#!/bin/bash

jupyter nbconvert *.ipynb --to pdf --output-dir pdf

jupyter nbconvert "Start Here.ipynb" --to pdf 
