#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

int M = 200; // Height
int N = 200; // Width

#define BLOCK_SIZE 16

void generateImg(int witdh, int height, int *img) {
    int img_size = witdh * height;
    for (int i = 0; i < img_size; i++) {
        img[i] = 0;
    }
}

void drawSolidRectangle(int width, int height, int *img) {
    for (int i = 1; i < height - 3; ++i) {
        for (int j = 1; j < width - 3; ++j) {
            img[M * i + j] = 1;
        }
    }
    for (int i = 4; i < height - 3; ++i) {
        for (int j = 4; j < width - 3; ++j) {
            img[M * i + j] = 0;
        }
    }
}

void generateClc(int *img, int *clc, int size_of_filter, int witdh, int height) {
    int index = (size_of_filter - 1) / 2;
    for (int i = 0; i < witdh; i++) {
        for (int j = 0; j < height; j++) {
            int row = i;
            int col = j;
            int index_row = row + index;
            int index_col = col + index;
            clc[index_row + index_col * (witdh + 2 * index)] = img[row + col * witdh];
        }
    }
    /*
    printf("Image : \n");
    for (int i =0;i<height;i++){
    printf("\n");
    for (int j=0;j<witdh;j++){
    printf ("%d   ",img[i*witdh+j]);
    }
    }
            
    printf("\n\nImage with adding border : \n");   
    for (int i =0;i<height+2*index;i++){
    printf("\n");
    for (int j=0;j<witdh+2*index;j++){
    printf ("%d   ",clc[i*(witdh+2*index)+j]);
    }
    }
    */
}

void erosionCPU(int *res_cpu, int *clc, int witdh, int height, int size_of_filter) {
    int index = (size_of_filter - 1) / 2;
    int pixel;
    for (int i = index; i < height + index; i++) {
        for (int j = index; j < witdh + index; j++) {
            pixel = 1;
            for (int k = 0; k < size_of_filter; k++) {
                for (int l = 0; l < size_of_filter; l++) {
                    if (pixel > clc[j - index + k + (i - index + l) * (witdh + 2 * index)]) {
                        pixel = clc[j - index + k + (i - index + l) * (witdh + 2 * index)];
                    }
                }
            }
            res_cpu[j - index + (i - index) * witdh] = pixel;
        }
    }
}

__global__ void erosionImg(int *res, int *clc, int witdh, int height, int size_of_filter) {

    int index = (size_of_filter - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;
    int pixel = 1;

    for (int i = 0; i < size_of_filter; i++) {
        for (int j = 0; j < size_of_filter; j++) {
            if (row < witdh + index && col < height + index) {
                int tmp = row + col * (witdh + index * 2);
                if (pixel > clc[row + (i - index) + (col + (j - index)) * (witdh + index * 2)]) {
                    pixel = clc[tmp + i - index + (j - index) * (witdh + index * 2)];
                }
            }

        }
    }

    if (row < witdh + index && col < height + index) {
        res[(col - index) * witdh + row - index] = pixel;
    }

}

// Function to perform erosion on GPU with shared memory
__global__ void erosionImgShared(int *res, int *clc, int witdh, int height, int size_of_filter) {
    __shared__ int clc_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // Shared memory for caching the filter region

    int index = (size_of_filter - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;

    // Copy data to shared memory
    clc_shared[threadIdx.x + 1][threadIdx.y + 1] = clc[row + (col * (witdh + 2 * index))];

    // Load boundary data to shared memory
    if (threadIdx.x < index) {
        clc_shared[threadIdx.x][threadIdx.y + 1] = clc[row - index + (col * (witdh + 2 * index))];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y + 1] = clc[row + BLOCK_SIZE + 1 + (col * (witdh + 2 * index))];
    }

    if (threadIdx.y < index) {
        clc_shared[threadIdx.x + 1][threadIdx.y] = clc[row + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + 1][threadIdx.y + BLOCK_SIZE + 1] = clc[row + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
    }

    // Load corner data to shared memory
    if (threadIdx.x < index && threadIdx.y < index) {
        clc_shared[threadIdx.x][threadIdx.y] = clc[row - index + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y] = clc[row + BLOCK_SIZE + 1 + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x][threadIdx.y + BLOCK_SIZE + 1] = clc[row - index + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y + BLOCK_SIZE + 1] =
            clc[row + BLOCK_SIZE + 1 + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    int pixel = 1;

    for (int i = 0; i < size_of_filter; i++) {
        for (int j = 0; j < size_of_filter; j++) {
            if (pixel > clc_shared[threadIdx.x + 1 + i - index][threadIdx.y + 1 + j - index]) {
                pixel = clc_shared[threadIdx.x + 1 + i - index][threadIdx.y + 1 + j - index];
            }
        }
    }

    if (row < witdh + index && col < height) {
        res[(col - index) * witdh + row - index] = pixel;
    }
}

void dilationCPU(int *res_cpu, int *clc, int witdh, int height, int size_of_filter) {
    int index = (size_of_filter - 1) / 2;
    int pixel;
    for (int i = index; i < height + index; i++) {
        for (int j = index; j < witdh + index; j++) {
            pixel = 0;
            for (int k = 0; k < size_of_filter; k++) {
                for (int l = 0; l < size_of_filter; l++) {
                    if (pixel < clc[j - index + k + (i - index + l) * (witdh + 2 * index)]) {
                        pixel = clc[j - index + k + (i - index + l) * (witdh + 2 * index)];
                    }
                }
            }
            res_cpu[j - index + (i - index) * witdh] = pixel;
        }
    }
}

__global__ void dilationImg(int *res, int *clc, int witdh, int height, int size_of_filter) {

    int index = (size_of_filter - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;
    int pixel = 0;


    for (int i = 0; i < size_of_filter; i++) {
        for (int j = 0; j < size_of_filter; j++) {
            if (row < witdh + index && col < height) {
                int tmp = row + col * (witdh + index * 2);
                if (pixel < clc[row + (i - index) + (col + (j - index)) * (witdh + index * 2)]) {
                    pixel = clc[tmp + i - index + (j - index) * (witdh + index * 2)];
                }
            }

        }
    }

    if (row < witdh + index && col < height) {
        res[(col - index) * witdh + row - index] = pixel;
    }

}
// Function to perform erosion on GPU with shared memory
__global__ void dilationImgShared(int *res, int *clc, int witdh, int height, int size_of_filter) {
    __shared__ int clc_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // Shared memory for caching the filter region

    int index = (size_of_filter - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;

    // Copy data to shared memory
    clc_shared[threadIdx.x + 1][threadIdx.y + 1] = clc[row + (col * (witdh + 2 * index))];

    // Load boundary data to shared memory
    if (threadIdx.x < index) {
        clc_shared[threadIdx.x][threadIdx.y + 1] = clc[row - index + (col * (witdh + 2 * index))];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y + 1] = clc[row + BLOCK_SIZE + 1 + (col * (witdh + 2 * index))];
    }

    if (threadIdx.y < index) {
        clc_shared[threadIdx.x + 1][threadIdx.y] = clc[row + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + 1][threadIdx.y + BLOCK_SIZE + 1] = clc[row + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
    }

    // Load corner data to shared memory
    if (threadIdx.x < index && threadIdx.y < index) {
        clc_shared[threadIdx.x][threadIdx.y] = clc[row - index + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y] = clc[row + BLOCK_SIZE + 1 + (col - index) * (witdh + 2 * index)];
        clc_shared[threadIdx.x][threadIdx.y + BLOCK_SIZE + 1] = clc[row - index + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
        clc_shared[threadIdx.x + BLOCK_SIZE + 1][threadIdx.y + BLOCK_SIZE + 1] =
            clc[row + BLOCK_SIZE + 1 + (col + BLOCK_SIZE + 1) * (witdh + 2 * index)];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    int pixel = 0;

    for (int i = 0; i < size_of_filter; i++) {
        for (int j = 0; j < size_of_filter; j++) {
            if (pixel < clc_shared[threadIdx.x + 1 + i - index][threadIdx.y + 1 + j - index]) {
                pixel = clc_shared[threadIdx.x + 1 + i - index][threadIdx.y + 1 + j - index];
            }
        }
    }

    if (row < witdh + index && col < height) {
        res[(col - index) * witdh + row - index] = pixel;
    }
}

void blurCPU(int *res_cpu, int *img, int witdh, int height, int size_of_filter) {
    int index = (size_of_filter - 1) / 2;
    for (int i = index; i < height + index; i++) {
        for (int j = index; j < witdh + index; j++) {
            int sum = 0;
            for (int k = 0; k < size_of_filter; k++) {
                for (int l = 0; l < size_of_filter; l++) {
                    sum += img[j - index + k + (i - index + l) * witdh];
                }
            }
            res_cpu[j - index + (i - index) * witdh] = sum / (size_of_filter * size_of_filter);
        }
    }
}

__global__ void blurImg(int *res, int *img, int witdh, int height, int size_of_filter) {
    int index = (size_of_filter - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;
    int sum = 0;

    for (int i = 0; i < size_of_filter; i++) {
        for (int j = 0; j < size_of_filter; j++) {
            if (row < witdh + index && col < height + index) {
                int tmp = row + col * (witdh + index * 2);
                sum += img[tmp + i - index + (j - index) * (witdh + index * 2)];
            }
        }
    }

    if (row < witdh + index && col < height + index) {
        res[(col - index) * witdh + row - index] = sum / (size_of_filter * size_of_filter);
    }
}

void sharpenCPU(int *res_cpu, int *img, int witdh, int height, float weight_center, float weight_neighbors) {
    int index = 1;  // Assuming a 3x3 sharpening filter
    for (int i = index; i < height - index; i++) {
        for (int j = index; j < witdh - index; j++) {
            int center = img[j + i * witdh];
            int neighbors = 0;

            // Calculate the weighted sum of neighboring pixels
            for (int k = -index; k <= index; k++) {
                for (int l = -index; l <= index; l++) {
                    if (k == 0 && l == 0) {
                        neighbors += weight_center * img[j + l + (i + k) * witdh];
                    } else {
                        neighbors += weight_neighbors * img[j + l + (i + k) * witdh];
                    }
                }
            }

            // Subtract the weighted sum from the center pixel value
            res_cpu[j - index + (i - index) * witdh] = center - neighbors;
        }
    }
}

__global__ void sharpenImg(int *res, int *img, int witdh, int height, float weight_center, float weight_neighbors) {
    int index = 1;  // Assuming a 3x3 sharpening filter
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;

    if (row < height - index && col < witdh - index) {
        int center = img[col + row * witdh];
        int neighbors = 0;

        // Calculate the weighted sum of neighboring pixels
        for (int i = -index; i <= index; i++) {
            for (int j = -index; j <= index; j++) {
                if (i == 0 && j == 0) {
                    neighbors += weight_center * img[col + j + (row + i) * witdh];
                } else {
                    neighbors += weight_neighbors * img[col + j + (row + i) * witdh];
                }
            }
        }

        // Subtract the weighted sum from the center pixel value
        res[(col - index) + (row - index) * witdh] = center - neighbors;
    }
}

__global__ void sharpenImgOptimized(int *res, int *img, int width, int height, float weight_center, float weight_neighbors) {
    int index = 1;  // Assuming a 3x3 sharpening filter
    int row = blockIdx.x * blockDim.x + threadIdx.x + index;
    int col = blockIdx.y * blockDim.y + threadIdx.y + index;

    __shared__ float sharedImg[18][18];

    int shared_row = threadIdx.x + index;
    int shared_col = threadIdx.y + index;

    if (row < height && col < width) {
        sharedImg[shared_row][shared_col] = img[row * width + col];

        // Load additional pixels to shared memory for border handling
        if (threadIdx.x < index) {
            sharedImg[threadIdx.x][shared_col] = img[row * width + col - index];
            sharedImg[threadIdx.x + blockDim.x + index][shared_col] = img[row * width + col + blockDim.x];
        }

        if (threadIdx.y < index) {
            sharedImg[shared_row][threadIdx.y] = img[(row - index) * width + col];
            sharedImg[shared_row][threadIdx.y + blockDim.y + index] = img[(row + blockDim.y) * width + col];
        }

        // Load additional pixels to shared memory for corner handling
        if (threadIdx.x < index && threadIdx.y < index) {
            sharedImg[threadIdx.x][threadIdx.y] = img[(row - index) * width + col - index];
            sharedImg[threadIdx.x + blockDim.x + index][threadIdx.y] = img[(row + blockDim.x) * width + col - index];
            sharedImg[threadIdx.x][threadIdx.y + blockDim.y + index] = img[(row - index) * width + col + blockDim.y];
            sharedImg[threadIdx.x + blockDim.x + index][threadIdx.y + blockDim.y + index] = img[(row + blockDim.x) * width + col + blockDim.y];
        }
    }

    __syncthreads();

    if (row < height - index && col < width - index) {
        int center = sharedImg[shared_row][shared_col];
        int neighbors = 0;

        // Calculate the weighted sum of neighboring pixels
        for (int i = -index; i <= index; i++) {
            for (int j = -index; j <= index; j++) {
                if (i == 0 && j == 0) {
                    neighbors += weight_center * sharedImg[shared_row][shared_col];
                } else {
                    neighbors += weight_neighbors * sharedImg[shared_row + i][shared_col + j];
                }
            }
        }

        // Subtract the weighted sum from the center pixel value
        res[(col - index) + (row - index) * width] = center - neighbors;
    }
}


int main()
{
    // Assume that the image is black and white.

    int witdh = N;
    int height = M;

    // Filter can only be an odd number
    int size_of_filter = 3;

    // Prepare the size of the matrix who'll help us to do the calculation
    int clc_witdh = witdh + size_of_filter - 1;
    int clc_height = height + size_of_filter - 1;

    int size = witdh * height * sizeof(int);
    int clc_size = clc_witdh * clc_height * sizeof(int);
    int *img;
    int *res;
    int *res_cpu;
    int *clc;
    cudaMallocManaged(&img, size);
    cudaMallocManaged(&res, size);
    cudaMallocManaged(&res_cpu, size);
    cudaMallocManaged(&clc, clc_size);

    float weight_center = 8.0;    // Weight for the center pixel
    float weight_neighbors = -1.0;  // Weight for the neighboring pixels

    int img_size = witdh * height;

    // TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // Call function to create the image;
    generateImg(witdh, height, img);
    drawSolidRectangle(256, 256, img);
    generateClc(img, clc, size_of_filter, witdh, height);

    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((witdh / threads_per_block.x) + 1, (height / threads_per_block.y) + 1, 1);

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    cudaEventRecord(start);

    erosionImg<<<number_of_blocks, threads_per_block>>>(res, clc, witdh, height, size_of_filter);

    cudaDeviceSynchronize();
    
    // Print TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Erosion GPU execution time: %f ms\n", elapsedTime);

    clock_t startCPU, stopCPU;
    double elapsedTimeCPU;

    startCPU = clock();

    // Perform erosion on CPU
    erosionCPU(res_cpu, clc, witdh, height, size_of_filter);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Erosion CPU execution time: %f ms\n", elapsedTimeCPU);

    for (int i = 0; i < N * M; i++) {
        if (res[i] != res_cpu[i]) {
            printf("res = %d res_cpu = %d ____ i value = %d\n", res[i], res_cpu[i], i);
        }
    }
    
    /*
    printf("\n\n GPU erosion: \n"); 
    for (int i = 0; i < height; i++) {
        printf("\n");
        for (int j = 0; j < witdh; j++) {
            printf("%d   ", res[i * witdh + j]);
        }
    }
    printf("\n\n CPU erosion: \n");
    for (int i = 0; i < height; i++) {
        printf("\n");
        for (int j = 0; j < witdh; j++) {
            printf("%d   ", res_cpu[i * witdh + j]);
        }
    }
    */

    cudaEventRecord(start);
    dilationImg<<<number_of_blocks, threads_per_block>>>(res, clc, witdh, height, size_of_filter);
    cudaDeviceSynchronize();

    // Print TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Dilation GPU execution time: %f ms\n", elapsedTime);

    startCPU = clock();

    // Perform dilation on CPU
    dilationCPU(res_cpu, clc, witdh, height, size_of_filter);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Dilation CPU execution time: %f ms\n", elapsedTimeCPU);

    for (int i = 0; i < N * M; i++) {
        if (res[i] != res_cpu[i]) {
            printf("res = %d res_cpu = %d ____ i value = %d\n", res[i], res_cpu[i], i);
        }
    }

    /*
    printf("\n\n GPU dilation: \n");
    for (int i = 0; i < height; i++) {
        printf("\n");
        for (int j = 0; j < witdh; j++) {
            printf("%d   ", res[i * witdh + j]);
        }
    }
    printf("\n\n CPU dilation: \n");
    for (int i = 0; i < height; i++) {
        printf("\n");
        for (int j = 0; j < witdh; j++) {
            printf("%d   ", res_cpu[i * witdh + j]);
        }
    }
    */

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEventRecord(start);

    // Perform erosion on GPU with shared memory
    erosionImgShared<<<number_of_blocks, threads_per_block, 0, stream1>>>(res, clc, witdh, height, size_of_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Erosion GPU (shared memory) execution time: %f ms\n", elapsedTime);

    // Additional operations can be overlapped with the erosion operation using streams

    // Perform dilation on GPU
    dilationImgShared<<<number_of_blocks, threads_per_block, 0, stream2>>>(res, clc, witdh, height, size_of_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Dilation GPU (shared memory) execution time: %f ms\n", elapsedTime);

    // Wait for streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // CPU Opening
    startCPU = clock();

    // Perform erosion on CPU
    erosionCPU(res_cpu, clc, witdh, height, size_of_filter);

    // Perform dilation on CPU
    dilationCPU(res_cpu, clc, witdh, height, size_of_filter);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Opening CPU execution time: %f ms\n", elapsedTimeCPU);

    // GPU Opening
    cudaEventRecord(start);
    erosionImg<<<number_of_blocks, threads_per_block>>>(res, clc, witdh, height, size_of_filter);
    dilationImg<<<number_of_blocks, threads_per_block>>>(res, clc, witdh, height, size_of_filter);
    cudaDeviceSynchronize();

    //print TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Opening GPU execution time: %f ms\n", elapsedTime);

    //Shared Opening
    // Create CUDA streams
    cudaStream_t stream3, stream4;
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEventRecord(start);

    // Perform erosion on GPU with shared memory
    erosionImgShared<<<number_of_blocks, threads_per_block, 0, stream3>>>(res, clc, witdh, height, size_of_filter);

    // Perform dilation on GPU
    dilationImgShared<<<number_of_blocks, threads_per_block, 0, stream4>>>(res, clc, witdh, height, size_of_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Opening GPU (shared memory) execution time: %f ms\n", elapsedTime);

    // Wait for streams to finish
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);

    //CPU Closing
    startCPU = clock();

    // Perform dilation on CPU
    dilationCPU(res_cpu, clc, witdh, height, size_of_filter);

    // Perform erosion on CPU
    erosionCPU(res_cpu, clc, witdh, height, size_of_filter);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Closing CPU execution time: %f ms\n", elapsedTimeCPU);

    //GPU Closing
    cudaEventRecord(start);
    dilationImg<<<number_of_blocks, threads_per_block>>>(res,clc, witdh, height,size_of_filter);
    erosionImg<<<number_of_blocks, threads_per_block>>>(res,clc, witdh, height,size_of_filter);
    cudaDeviceSynchronize();
    
    
    //print TIMER
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Closing GPU execution time: %f ms\n", elapsedTime);

    // Shared Closing
    // Create CUDA streams
    cudaStream_t stream5, stream6;
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);

    cudaEventRecord(start);
    // Perform dilation on GPU
    dilationImgShared<<<number_of_blocks, threads_per_block, 0, stream6>>>(res, clc, witdh, height, size_of_filter);

    // Perform erosion on GPU with shared memory
    erosionImgShared<<<number_of_blocks, threads_per_block, 0, stream5>>>(res, clc, witdh, height, size_of_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Closing GPU (shared memory) execution time: %f ms\n", elapsedTime);

    // Wait for streams to finish
    cudaStreamSynchronize(stream5);
    cudaStreamSynchronize(stream6);

    // Sharpen
    startCPU = clock();

    // Sharpen on CPU
    sharpenCPU(res_cpu, img, witdh, height, weight_center, weight_neighbors);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Sharpen CPU execution time: %f ms\n", elapsedTimeCPU);

    cudaEventRecord(start);
    // Sharpen on GPU
    sharpenImgOptimized<<<number_of_blocks, threads_per_block>>>(res, img, witdh, height, weight_center, weight_neighbors);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Sharpen Optimize GPU execution time: %f ms\n", elapsedTime);

    cudaEventRecord(start);
    // Sharpen on GPU
    sharpenImg<<<number_of_blocks, threads_per_block>>>(res, img, witdh, height, weight_center, weight_neighbors);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Sharpen GPU  execution time: %f ms\n", elapsedTime);

    /*
    // Check for errors (all values should be close enough)
    for (int i = 0; i < img_size; i++) {
        if (abs(res_cpu[i] - res[i]) > 1) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, res_cpu[i], res[i]);
        break;
        }
    }
    */
    // Blur
    startCPU = clock();

    // Blur on CPU
    //blurCPU(res_cpu, img, witdh, height, size_of_filter);

    stopCPU = clock();
    elapsedTimeCPU = ((double)(stopCPU - startCPU)) * 1000.0 / CLOCKS_PER_SEC;
    printf("\n\n Blur CPU execution time: %f ms\n", elapsedTimeCPU);

    cudaEventRecord(start);
    // Blur on GPU
    blurImg<<<number_of_blocks, threads_per_block>>>(res, img, witdh, height, size_of_filter);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\n Blur GPU  execution time: %f ms\n", elapsedTime);

    /*
    // Check for errors (all values should be close enough)
    for (int i = 0; i < img_size; i++) {
        if (abs(res_cpu[i] - res[i]) > 1) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, res_cpu[i], res[i]);
        break;
        }
    }
    */
    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    cudaFree(img);
    cudaFree(res);
    cudaFree(res_cpu);
    cudaFree(clc);
}
;
