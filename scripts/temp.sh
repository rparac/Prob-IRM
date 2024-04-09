#!/bin/bash


for dir in */; do
  echo $dir
  cd $dir
  for dir2 in */; do
    mv $dir2/* .
    rm -rf $dir2
  done
  cd ..
done