# Active Oasis

This is an experimental tool to find associative and commutative permutations and reduce of elliptic curve operations using term rewriting. Given an input statement it aims to rewrite it into all possible variations which will provide the same result, for example:

```
Expr: G(Add(w,Mul(Hp(G(preimage),G(w)),preimage)))
Vars: preimage, w

= Point(Add(w,Mul(Hp(Point(preimage),Point(w)),preimage)))
= Point(Add(Mul(Hp(Point(preimage),Point(w)),preimage),w))
= PointAdd(Point(w),ScalarMult(Point(Hp(Point(preimage),Point(w))),preimage))
= PointAdd(Point(w),ScalarMult(Point(preimage),Hp(Point(preimage),Point(w))))
= Point(Add(Mul(preimage,Hp(Point(preimage),Point(w))),w))
= Point(Add(w,Mul(preimage,Hp(Point(preimage),Point(w)))))
= PointAdd(ScalarMult(Point(Hp(Point(preimage),Point(w))),preimage),Point(w))
= PointAdd(ScalarMult(Point(preimage),Hp(Point(preimage),Point(w))),Point(w))
```

This may useful in the following scenarios:

 * Security testing
 * Informal verification
 * Protocol design
 * Complexity reduction
 * Optimisation

It uses the [trstools.py](https://github.com/obfusk/trstools.py) term rewriting library by [Felix C. Stegerman](flx@obfusk.net) with some minor modifications, and the [py_ecc](https://github.com/ethereum/py_ecc) library by [Vitalik Buterin](https://vitalik.ca) for the BN256 curve implementation.

## Interested?

I am open to insights about extra things which can be added using simple term rewrites, e.g.:

 * Relationships between integers
 * Simplifications
 * Logical symmetries
 * Curve features (division, inversion etc.)

Feel free to create an Issue if you think of anything, even adding very basic relationships between integers can greatly improve this tools ability to find interesting or unexpected solutions.

e.g. think of simplifying operations like `(x*2)-x = x`
