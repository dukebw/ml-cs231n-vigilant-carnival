#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#define NUM_LABELS_PER_BATCH 10000
#define NUM_TRAIN 5000
#define NUM_TEST 500
#define IMAGE_DIM 32
#define BYTES_PER_IMAGE (IMAGE_DIM*IMAGE_DIM*3)
#define UINT64_PER_IMAGE (BYTES_PER_IMAGE/sizeof(uint64_t))

static int32_t global_square_distance[NUM_TEST][NUM_TRAIN];

int main(void)
{
        int32_t data_fd = open("cifar-10-batches-bin/data_batch_1.bin",
                               O_RDONLY);
        assert(data_fd != -1);

        struct stat sb;
        int32_t status = fstat(data_fd, &sb);
        assert(status != -1);

        uint64_t *x_train = mmap(NULL,
                                 sb.st_size,
                                 PROT_READ,
                                 MAP_PRIVATE,
                                 data_fd,
                                 0);
        assert(x_train != MAP_FAILED);

        uint64_t *x_test = (uint64_t *)((uint8_t *)x_train + NUM_TRAIN*(BYTES_PER_IMAGE + 1));

#pragma omp parallel for
        for (uint32_t test_index = 0;
             test_index < NUM_TEST;
             ++test_index) {
                for (uint32_t train_index = 0;
                     train_index < NUM_TRAIN;
                     ++train_index) {
                        for (uint32_t i = 0;
                             i < UINT64_PER_IMAGE;
                             ++i) {
                                int32_t delta = (x_train[UINT64_PER_IMAGE*train_index + i] -
                                                 x_test[UINT64_PER_IMAGE*test_index + i]);
                                global_square_distance[test_index][train_index] += delta*delta;
                        }
                }
        }
}
