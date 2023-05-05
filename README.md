# multiprocessor

how to build:

```make build mainfile=<filename>```

replace \<filename\> with the file you want to use

for example for phase4 (OpenCl implementation):

```make build mainfile=phase4```


directories:
- src: C source code
- inc: C headers
- kernels: OpenCL kernels
- images: input images

phases of the project:
- src/phase2.c
  - C implementation
- src/phase3.c
  - C implementation with OpenMP
- src/phase4.c
  - OpenCL implementation


running (phase2/3/4):

```./main "images/im0.png" "images/im1.png" "output.png"```
