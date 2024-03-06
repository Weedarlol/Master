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


# Content Infomration

"fox" and "ex3" are two different clusters i worked on which i was allowed to use to test my projects. The folders are therefore quite similar, the biggest difference is that the folders are specialised for scenarios which i was allowed to run in the specified cluster

## Fox cluster
The fox cluster was provided to me by UiO through Educloud Research

Fox had 2 partitions i was utilizing, "Normal" and "Accel"
- Normal was made up

## Ex3 Cluster
The Ex3 cluster was provided to me by Simula through my project guidance Xing Cai.

Ex3 had 3 partitions i was utilizing, "Dgx2q", "Hgx2q" and "A100q". 
- Dgx2q was made up of a single node consisting of DualProcessor Intel Xeon Scalable Platinum 8176 with 16 Nvidia Volta V100 connected through PCIe, but which are interconnected by NVLink.
- Hgx2q was made up of a single node consisting of DualProcessor AMD EPYC Milan 7763 64-core with 8 Nvidia Volta A100/80GB connected through PCIE, but which are interconnected by NVLink.
- A100q


