#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#define NUM_LABELS_PER_BATCH 10000
#define IMAGE_DIM 32
#define BYTES_PER_IMAGE (IMAGE_DIM*IMAGE_DIM*3)
#define BYTES_PER_DATUM (BYTES_PER_IMAGE + 1)

int main(void)
{
        int32_t status;
        FILE *data_handle = fopen("cifar-10-batches-bin/data_batch_1.bin",
                                  "rb");

        for (uint32_t label_index = 0;
             label_index < NUM_LABELS_PER_BATCH;
             ++label_index) {
                printf("label %d: %d\n", label_index, fgetc(data_handle));

                status = fseek(data_handle, BYTES_PER_IMAGE, SEEK_CUR);
                assert(status == 0);
        }

        fclose(data_handle);
}
