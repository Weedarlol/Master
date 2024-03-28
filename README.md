# Project Title

**Aiming at Heterogeneous Parallel Computing for Bioinformatics: To Overlap or Not to Overlap?**

## Keywords

- Heterogeneous Computing
- Parallel Programming
- Computational Bioinformatics

## Introduction

The need for more computing power applies to all subjects of computational science. This is no exception in the case of computational bioinformatics, where typical applications that require supercomputing power include DNA sequence analytics and molecular dynamics simulations. Another timely example is the modeling of the spread of viruses.

One way to answer the urgent need for supercomputing is to adopt heterogeneous computing platforms, where a system has at least two types of processor architectures. A typical configuration is a cluster of compute nodes where each node consists of a host CPU and one or several GPUs (graphics processing units). One of the research questions related to parallel programming of such heterogeneous systems is whether to overlap communication with computation for the purpose of "hiding" the communication overhead. In particular, the appropriateness of overlap in the era of heterogeneous computing requires new research. This is both due to the extra programming complexity that will be induced by implementing communication-computation overlap and due to the impact of, for example, memory bandwidth contention, which may lead to a slower computation speed as a whole.

## Goal

This master's degree project aims at a detailed and quantifiable understanding of the impact of overlapping communication with computation on heterogeneous parallel computers. The candidate will start with developing simple synthetic benchmark programs that implement various scenarios of communication-computation overlap. These benchmarks will be carefully experimented and time-measured for understanding the potential performance benefits (or disadvantages). An effort will be made to establish performance models related to communication-computation overlap or non-overlap.

Another objective of this master's degree project is to summarize some programming guidelines in this regard. The scientific findings will then be tested in applications that fall within the domain of computational bioinformatics to check the effectiveness of the resulting heterogeneous programming approach (with or without communication-computation overlap), as well as the applicability of the performance modeling methodology.

## Learning Outcome

The candidate will learn about advanced parallel programming applicable to both multicore CPUs and at least one specific GPU architecture. The candidate will also become an expert on performance profiling and modeling. The candidate will have the chance to become familiarized with real-world examples of computational bioinformatics. The candidate will also be exposed to cutting-edge hardware for parallel computing. These skills and experiences are highly sought-after expertise for the future workforce in scientific and technical computing.

## Qualifications Required

The candidate is expected to be skillful in technical programming (experience with parallel programming is not required but preferred). Importantly, the candidate must be hardworking and eager to learn new skills and knowledge, such as basic mathematical modeling and basic bioinformatics applications.



# Content Information

For running my tests i was given access to two different clusters. The first cluster was named Fox and was provided to me through Educloud Research, while the other was eX3 which was provided through Simula Research Laboratory.

These two clusters was chosen as they contained nodes which contained both CPUs and GPUs which could communicate with eachother through either PCIe or NVLink.

The folders "ex3" and "fox" are therefore each assigned to their own cluster, but they are in practice quite similar in code.

## Fox cluster
The fox cluster was provided to me by Educloud Research through my masterstudent status at University of Oslo

Fox had several nodes i was utilizing called "Accel" nodes, each of these nodes consisted of the following parts
- AMD EPYC 7702 64-Core Processor CPU
- 4x NVIDIA A100 GPUs

## Ex3 Cluster
The Ex3 cluster was provided to me by Simula Research Laboratory through my project guidance Xing Cai.

Ex3 had 3 partitions i was utilizing, "Dgx2q" and "Hgx2q". Each corresponding to only 1 node.

Dgx2q was the less powerfull of the two partitions and it was made up of an [Dgx2](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-1/dgx-2-datasheet-us-nvidia-955420-r2-web-new.pdf) machine, but with the following parts
- DualProcessor Intel Xeon Scalable Platinum 8176 CPU
- 16x Nvidia V100 GPUs

Hgx2q is the more powerfull partition and is also made up of its own machine called [Hgx2](https://images.nvidia.com/content/pdf/hgx2-datasheet.pdf), but with the following parts
- DualProcessor AMD EPYC Milan 7763 64-core CPU
- 8x Nvidia A100/80GB GPUs

# Tasks
## CPU
- [x] CPU 2d
- [x] CPU 3d

## GPU
- [x] GPU 2d 1 GPU
- [x] GPU 2d
- [x] GPU 3d 1 GPU
- [ ] GPU 3d

## Nodes
- [x] CPU to CPU
- [ ] CPU to GPU
- [ ] GPU to GPU

## Plots
- [x] 2d plots
- [x] 3d plots
- [ ] Communication plot
- [ ] Node plot

## General
- [ ] Create one folder for all clusters
- [ ] Clean up github
- [ ] Finish tests
- [ ] Finish Master Document