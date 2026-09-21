extern "C" __global__ void gather_tile_weights(
    const float* weights, const int* ids, float* output,
    int tiles, int D, int K, int N, int id_stride) {
    long long total=(long long)tiles*D*K;
    for(long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        index<total;index+=(long long)blockDim.x*gridDim.x) {
        int slot=index%K, d=(index/K)%D, tile=index/(K*D);
        output[index]=weights[d*N+ids[tile*id_stride+slot]];
    }
}


extern "C" __global__ void tile_layout(
    const float* input, const int* voxel_ids, const long long* order, float* output,
    int tiles, int D, int M, int V, int offset, int reverse) {
    long long total=(long long)tiles*D*M;
    for(long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        index<total;index+=(long long)blockDim.x*gridDim.x) {
        int v=index%M, d=(index/M)%D, tile=index/(M*D);
        int voxel=voxel_ids[order[tile+offset]*M+v];
        if(reverse) output[index]=voxel>=0?input[(long long)d*V+voxel]:0.f;
        else if(voxel>=0) output[(long long)d*V+voxel]=input[index];
    }
}


extern "C" __global__ void tile_weight_backward(
    const float* upstream, const int* inverse, float* gradient,
    int tiles, int K, int D, int N) {
    int j=blockIdx.x, tid=threadIdx.x, lane=tid%32, warp=tid/32;
    __shared__ float partial[256];
    for(int base=0;base<D;base+=32) {
        int f=base+lane;
        float value=0;
        if(f<D) for(int tile=warp;tile<tiles;tile+=8) {
            int slot=inverse[j*tiles+tile];
            if(slot>=0) value+=upstream[((long long)tile*K+slot)*D+f];
        }
        partial[tid]=value;
        __syncthreads();
        if(warp==0 && f<D) {
            float sum=0;
            for(int w=0;w<8;w++) sum+=partial[32*w+lane];
            gradient[f*N+j]=sum;
        }
        __syncthreads();
    }
}


extern "C" __global__ void tile_candidates(
    const float* lo, const float* hi, const float* centers, const float* extents,
    int* ids, int* counts, int* inverse, int N, int tiles) {
    int tile=blockIdx.x, tid=threadIdx.x, lane=tid%32, warp=tid/32;
    __shared__ int warp_counts[8];
    __shared__ int base;
    if(tid==0) base=0;
    __syncthreads();
    for(int start=0;start<N;start+=256) {
        int j=start+tid;
        bool hit=j<N;
        if(hit) for(int d=0;d<3;d++)
            hit=hit && (centers[3*j+d]+extents[3*j+d]>=lo[3*tile+d])
                    && (centers[3*j+d]-extents[3*j+d]<=hi[3*tile+d]);
        unsigned mask=__ballot_sync(0xffffffff,hit);
        if(lane==0) warp_counts[warp]=__popc(mask);
        __syncthreads();
        int offset=0;
        for(int w=0;w<warp;w++) offset+=warp_counts[w];
        unsigned before=lane==0?0:((1u<<lane)-1u);
        if(hit) {
            int slot=base+offset+__popc(mask&before);
            ids[tile*N+slot]=j;
            inverse[j*tiles+tile]=slot;
        }
        __syncthreads();
        if(tid==0) for(int w=0;w<8;w++) base+=warp_counts[w];
        __syncthreads();
    }
    if(tid==0) counts[tile]=base;
}

extern "C" __global__ void tile_basis_forward(
    const float* xyz, const int* valid, const int* ids, const int* counts,
    const float* centers, const float* scales, const float* rotation,
    float* basis, int tiles, int K, int N, int M) {
    long long total=(long long)tiles*K*M;
    for(long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        index<total;index+=(long long)gridDim.x*blockDim.x) {
        int v=index%M, slot=(index/M)%K, tile=index/(M*K);
        if(slot>=counts[tile] || !valid[tile*M+v]) {basis[index]=0;continue;}
        int j=ids[tile*N+slot];
        float d[3],u[3]={0,0,0};
        for(int a=0;a<3;a++) d[a]=xyz[(tile*M+v)*3+a]-centers[3*j+a];
        float q=0;
        for(int a=0;a<3;a++) {
            for(int b=0;b<3;b++) u[a]+=d[b]*rotation[j*9+b*3+a];
            float s=scales[j*3+a];q+=u[a]*u[a]/(s*s);
        }
        basis[index]=__expf(-.5f*q);
    }
}


extern "C" __global__ void tile_basis_backward(
    const float* xyz, const int* valid, const int* inverse,
    const float* centers, const float* scales, const float* rotation,
    const float* basis, const float* upstream,
    float* gc, float* gs, float* gr, int K, int N, int M, int tiles) {
    int j=blockIdx.x, tid=threadIdx.x;
    float accum[15]={0};
    for(int tile=0;tile<tiles;tile++) {
      int slot=inverse[j*tiles+tile];
      if(slot<0) continue;
    for(int v=tid;v<M;v+=blockDim.x) {
        if(!valid[tile*M+v]) continue;
        long long index=((long long)tile*K+slot)*M+v;
        float d[3],u[3]={0,0,0},du[3];
        for(int a=0;a<3;a++) d[a]=xyz[(tile*M+v)*3+a]-centers[3*j+a];
        float h=upstream[index]*basis[index];
        for(int a=0;a<3;a++) {
            for(int b=0;b<3;b++) u[a]+=d[b]*rotation[j*9+b*3+a];
            float s=scales[j*3+a];
            du[a]=-h*u[a]/(s*s);
            accum[3+a]+=h*u[a]*u[a]/(s*s*s);
        }
        for(int b=0;b<3;b++) for(int a=0;a<3;a++) {
            accum[b]-=du[a]*rotation[j*9+b*3+a];
            accum[6+b*3+a]+=d[b]*du[a];
        }
    }
    }

    __shared__ float partial[15*8];
    int lane=tid%32, warp=tid/32;
    for(int a=0;a<15;a++) {
        float value=accum[a];
        for(int s=16;s>0;s/=2) value+=__shfl_down_sync(0xffffffff,value,s);
        if(lane==0) partial[a*8+warp]=value;
    }
    __syncthreads();
    if(tid<15) {
        float value=0;
        for(int w=0;w<blockDim.x/32;w++) value+=partial[tid*8+w];
        if(tid<3) gc[3*j+tid]=value;
        else if(tid<6) gs[3*j+tid-3]=value;
        else gr[9*j+tid-6]=value;
    }
}
