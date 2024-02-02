# wta
This repo contains code for running parallelized frugal resolvent splittings based on a list of resolvents and matrices for passing resolvents ($L$) and consensus ($W$) parameters.

The code is demonstrated on a weapon target assignment (WTA) problem. The WTA problem assigns weapons to targets so that the total post-attack expected value of the targets is minimized. The problem is formulated as a convex MIP and solved using CVXPY and MOSEK.

## Installation
```
$ git clone https://github.com/peterbarkley/wta.git && cd wta
$ pip install -r requirements.txt
```

I also recommend installing MOSEK for faster optimization. You can get a free academic license from the [MOSEK website] (https://www.mosek.com/products/academic-licenses/). 

## Usage - basic L1 norm and Huber mean examples
```
$ python test_main.py
```

## Usage - WTA example (requires MOSEK as default solver)
```
$ python test_wta.py 
```

## Notes
You need to build your own resolvent class(es) which can be initialized with your data at a given processor, and provides a shape attribute and a `__call__` method. The `__call__` method should return the resolvent at the given point.

There's a default set of matrices for any number of nodes using the Malitsky-Tam splitting algorithm, which will have minimal cycle time for splittings when the resolvents all take about the same time. Use `oars.solveMT(n, data, resolvents)`. There is also a four-splitting I built in the `test_wta.py` – its cycle time is optimal for situations where the first resolvent takes longer than the others. It also converges faster on the WTA problem than Malitsky-Tam. 