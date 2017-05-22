#include <omp.h>
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
#define BYTES_PER_EXAMPLE (BYTES_PER_IMAGE + 1)

static uint32_t global_square_distance[NUM_TEST][NUM_TRAIN];

static uint8_t
get_cifar10_img_pixel(uint8_t *data, uint32_t img_index, uint32_t pixel)
{
        return data[BYTES_PER_EXAMPLE*img_index + 1 + pixel];
}

int main(void)
{
        omp_set_num_threads(omp_get_num_procs());

        int32_t data_fd = open("cifar-10-batches-bin/data_batch_1.bin",
                               O_RDONLY);
        assert(data_fd != -1);

        struct stat sb;
        int32_t status = fstat(data_fd, &sb);
        assert(status != -1);

        uint8_t *x_train = mmap(NULL,
                                 sb.st_size,
                                 PROT_READ,
                                 MAP_PRIVATE,
                                 data_fd,
                                 0);
        assert(x_train != MAP_FAILED);

        uint8_t *x_test = x_train + NUM_TRAIN*BYTES_PER_EXAMPLE;

#pragma omp parallel for
        for (uint32_t test_index = 0;
             test_index < NUM_TEST;
             ++test_index) {
                for (uint32_t train_index = 0;
                     train_index < NUM_TRAIN;
                     ++train_index) {
                        for (uint32_t i = 0;
                             i < BYTES_PER_IMAGE;
                             ++i) {
                                int32_t delta = (get_cifar10_img_pixel(x_train, train_index, i) -
                                                 get_cifar10_img_pixel(x_test, test_index, i));
                                global_square_distance[test_index][train_index] += delta*delta;
                        }
                }
        }

        uint32_t num_correct = 0;
        for (uint32_t test_index = 0;
             test_index < NUM_TEST;
             ++test_index) {
                uint32_t nn_index = 0;
                uint32_t nn_distance = UINT32_MAX;
                for (uint32_t train_index = 0;
                     train_index < NUM_TRAIN;
                     ++train_index) {
                        if (global_square_distance[test_index][train_index] < nn_distance) {
                                nn_index = train_index;
                                nn_distance = global_square_distance[test_index][train_index];
                        }
                }

                if (x_train[nn_index*BYTES_PER_EXAMPLE] == x_test[test_index*BYTES_PER_EXAMPLE])
                        ++num_correct;
        }

        printf("k=1 %.2f\n", (float)num_correct/(float)NUM_TEST);
}
