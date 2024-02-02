# wta
This repo contains code for running parallelized frugal resolvent splittings based on a list of resolvents and matrices for passing resolvents ($L$) and consensus ($W$) parameters.

The code is demonstrated on a weapon target assignment (WTA) problem. The WTA problem is a combinatorial optimization problem that arises in the context of defense planning. The problem is to assign weapons to targets in such a way that the total post-attack expected value of targets to the targets is minimized. The problem is formulated as a convex MIP and solved using CVXPY and MOSEK.

## Installation
pip install -r requirements.txt

## Usage - basic L1 norm and Huber mean examples
python test_main.py

## Usage - WTA example
python test_wta.py