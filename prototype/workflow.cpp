/*
@author: Jiyuan Wang
@time: Jul 7, 2022
*/

#define AFL_MAIN
#define MESSAGES_TO_STDOUT

#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#ifdef __cplusplus
extern "C" {
#endif
#include "config.h"
#include "types.h"
#include "debug.h"
#include "alloc-inl.h"
#include "hash.h"

#include <io.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <dirent.h>
#include <ctype.h>
#include <fcntl.h>

#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#ifdef __cplusplus
}
#endif

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream> 


/* Main entry point */

int main(int argc, char** argv) {

    SAYF(cCYA "differential-testing-fuzz-on-devcloud " cBRI VERSION cRST " by <wangjiyuan@cs.ucla.edu>\n");

    if (argv[2]=="fpga"){
        std::string execute = "qsub -l nodes=s001-n085:ppn=2 " + argv[1];
    }
    else if (argv[2]=="gpu"){
        std::string execute = "qsub -l nodes=1:gpu:ppn=2 " + argv[1];
    }

    int a = std::system(execute.c_str());

    std::string saved_output = "prototype/good-outputs/*";;
    for (const auto & file : std::filesystem::directory_iterator(path))
    {
        if (argv[2]=="gpu"){
            std::string execute = "qsub -l nodes=s001-n085:ppn=2 " + file.path();
        }
        else if (argv[2]=="fpga"){
            std::string execute = "qsub -l nodes=1:gpu:ppn=2 " + file.path();
        }

    }

    return 0;
}