#!/bin/bash
cd ./code/model/correlation_package
./make.sh
cd ../../..
cd ./code/model/resample2d_package 
./make.sh
cd ../../..
cd ./code/model/channelnorm_package 
./make.sh
cd ../../..
