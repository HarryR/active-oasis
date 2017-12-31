# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import trstools as T
from py_ecc import bn128
from random import randint
from hashlib import sha256
from py_ecc.bn128 import add, multiply, double, curve_order, field_modulus, G1, eq
from py_ecc.bn128.bn128_field_elements import inv


bytes_to_int = lambda x: reduce(lambda o, b: (o << 8) + ord(b), [0] + list(x))
rands = lambda: randint(1, curve_order - 1)
sbmul = lambda s: multiply(G1, s)
hashs = lambda *x: bytes_to_int(sha256('.'.join(['%X' for _ in range(0, len(x))]) % x).digest()) % curve_order
hashp = lambda *x: hashs(*[item.n for sublist in x for item in sublist])


expr_funcs = dict(	
	G=sbmul,
	Point=lambda x: x if isinstance(x, tuple) else sbmul(x),
	PointInfinity=lambda: None,
	PointDouble=lambda x: double(x),
	Sub=lambda x, y: (x - y) % curve_order,
	Neg=lambda x: -x % curve_order,
	Inv=lambda x: inv(x, curve_order),
	#InvMul=lambda x, y: (x * pow(y, field_modulus-2, field_modulus)),
	Add=lambda x, y: (x + y) % curve_order,
	Mul=lambda x, y: (x * y) % curve_order,
	ScalarMult=multiply,
	PointNeg=lambda x: (x[0], -x[1]),
	PointAdd=add,
	Hs=hashs,
	Hp=hashp,
	Equal=lambda x, y: x == y,
)


def extract_vars(term):
	return {str(x): rands() for x in T.variables(term)}


def eval_expr(expr, local_vars):
	expr_env = expr_funcs.copy()
	expr_env.update(local_vars)
	return eval(expr, {'__builtins__': None}, expr_env)


rs = [
	# Double, half and add
	T.rule("Double(x) -> Add(x, x)"),
	T.rule("Half(Double(x)) -> x"),
	T.rule("Double(Half(x)) -> x"),

	# Square
	T.rule("Square(x) -> Add(Double(x), x)"),
	T.rule("Square(x) -> Add(x, Double(x))"),

	# Addition
	T.rule("Add(y, Double(x)) -> Add(Double(x), y)"),
	T.rule("Add(y, Double(y)) -> Add(Double(y), y)"),
	T.rule("Add(Mul(x, y),Mul(x,z)) -> Mul(x, Add(y,z))"),
	T.rule("PointAdd(ScalarMult(x, y), ScalarMult(x, z)) -> ScalarMult(x, Add(y, z))"),

	# Subtraction
	T.rule("Sub(Add(x, y), y) -> x"),
	T.rule("Sub(Add(x, y), x) -> y"),
	T.rule("Sub(x, Add(x, y)) -> x"),
	T.rule("Sub(y, Add(x, y)) -> y"),
	T.rule("Sub(Double(x), x) -> x"),

	# Multiplication
	T.rule("Mul(x, Mul(y, z)) -> Mul(Mul(x, y), z)"),
	T.rule("Mul(x, Half(y)) -> Half(Mul(x, y))"),
	T.rule("Mul(x, Double(y)) -> Double(Mul(x, y))"),

	# Inverse Multiplication
	T.rule("Mul(Mul(x, y), Inv(y)) -> x"),

	# Multiply by base point
	# `G(x)` is an opaque value
	T.rule("G(x) -> Point(x)"),

	# Discrete log...
	#T.rule("D(x) -> x"),
	#T.rule("D(Point(x)) -> x"),

	# Double wrapping of point
	T.rule("Point(Point(x)) -> Point(x)"),

	# Symmetry of Point Add
	T.rule("PointAdd(G(x), G(y)) -> PointAdd(Point(y), Point(x))"),
	T.rule("PointAdd(G(x), G(y)) -> PointAdd(Point(x), Point(y))"),

	# Point Add
	T.rule("PointAdd(G(x), G(y)) -> G(Add(x, y))"),
	T.rule("PointAdd(G(x), G(y)) -> G(Add(y, x))"),
	T.rule("PointAdd(G(y), G(x)) -> G(Add(x, y))"),
	T.rule("PointAdd(G(y), G(x)) -> G(Add(y, x))"),

	# Point Double
	T.rule("PointDouble(Point(x)) -> PointAdd(Point(x),Point(x))"),

	# PointAdd on opaque points remains opaque
	T.rule("PointAdd(Point(y), Point(x)) -> Point(Add(y, x))"),
	T.rule("PointAdd(Point(y), Point(x)) -> Point(Add(x, y))"),

	# Negation
	T.rule("Add(Add(x, x), Neg(x)) -> x"),
	T.rule("Neg(Neg(x)) -> x"),
	T.rule("PointNeg(PointNeg(Point(x))) -> Point(x)"),
	T.rule("PointNeg(Point(x)) -> Point(Neg(x))"),
	T.rule("G(Neg(x)) -> PointNeg(G(x))"),

	T.rule("PointAdd(Point(x), PointNeg(Point(x))) -> PointInfinity()"),
	# Subtraction via negation
	T.rule("PointAdd(Point(Add(s,s)),PointNeg(Point(s))) -> Point(s)"),

	# Point Multiply
	T.rule("ScalarMult(G(x), y) -> G(Mul(y, x))"),
	T.rule("ScalarMult(G(x), y) -> G(Mul(x, y))"),
	T.rule("ScalarMult(G(x), y) -> ScalarMult(G(y), x)"),

	# Multiplication on opaque points
	T.rule("ScalarMult(ScalarMult(Point(x), y), z) -> ScalarMult(Point(x), Mul(y, z))"),
	T.rule("ScalarMult(ScalarMult(Point(x), y), z) -> ScalarMult(Point(x), Mul(z, y))"),

	# Transform scalar addition to point addition
	T.rule("G(Add(x, y)) -> PointAdd(G(x), G(y))"),
	T.rule("G(Mul(x, y)) -> ScalarMult(G(x), y)"),
	T.rule("G(Mul(x, y)) -> ScalarMult(G(y), x)"),

]

if __name__ == "__main__":
	#for u,v in T.critical_pairs(rs):
	#	print("[ %-16s -> %-12s ]" % (u, v))

	#a = T.term("ScalarMult(G(x),y)")
	#b = T.term("ScalarMult(G(y), x)")
	#a = T.term("ScalarMult(G(x), y)")
	#a = T.term("ScalarMult(ScalarMult(Point(x), y), z)")
	#a = T.term("""ScalarMult(ScalarMult(PointAdd(Point(a), G(b)), c), z)""")
	#a = T.term("G(Add(w, Mul(p,z)))")
	#a = T.term("ScalarMult(PointAdd(G(a), Point(b)), c)")
	#a = T.term("G(Add(Hs(c), Mul(Mul(Add(a, b),c), z)))")
	#a = T.term("PointAdd(Point(Hs(c)),ScalarMult(Point(c),Mul(z,Add(b,a))))")
	#a = T.term("Equal(Point(s),PointNeg(PointNeg(Point(s))))")
	#a = T.term("Equal(Point(s), PointAdd(PointDouble(Point(s)),PointNeg(Point(s))) )")
	#a = T.term("Equal(PointDouble(Point(a)),PointAdd(Point(a),Point(a)))")
	#a = T.term("Equal(G(Neg(x)),PointNeg(G(x)))")

	# Example of schnorr protocol
	"""
	a = T.withvars(
		T.term("G(s)"), dict(
			b="G(preimage)",
			t="G(w)",
			c="Hp(b,t)",
			s="Add(w, Mul(c, preimage))",

			# The verifiers side is fixed by using 'Point' instead of 'G'
			# This means that they have the Point, resulting from the calculation
			# Rather than knowing the constituents of the calculation
			vb="Point(preimage)",
			vx="PointAdd(t, ScalarMult(vb, c))",
			vs="Point(s)",
		))
	"""

	# Another example of schnorr signature
	"""
	a = T.withvars(
		T.term("G(s)"), dict(
			y="Point(x)",
			r="G(k)",	# R = k*G
			e="Hs(m,Hp(r))",	# H(m||R)
			s="Add(k, Mul(x, e))",  # s = k + x*e
			svg="PointAdd(r, ScalarMult(y, e))", # svg = S.R + e*Y
			sg="Point(s)",
		))
	"""

	# Ring Signature over 2 keys
	a = T.withvars(
		T.term("verify"), dict(
			p0="Point(s0)",
			p1="G(s1)",

			# msgPoint = hashp
			m="G(msgHash)",

			# msgPointVerify = hashSP = Tau
			tau="Point(ScalarMult(m,s1))",

			signAcc="Hp(p0, p1, m, tau)",

			# Ring entry for public key 0
			a0="PointAdd(G(t0), ScalarMult(p0, c0))",
			b0="PointAdd(ScalarMult(m, t0), ScalarMult(tau, c0))",
			r0c="c0",
			r0t="t0",
			signAcc0="Hs(signAcc, Hp(a0, b0))",
			csum="c0",	# csum only updated if we don't hold secret

			# Ring entry for public key 1 (where we control secret key)
			a1="G(ri)",
			b1="ScalarMult(m, ri)",
			signAcc1="Hs(signAcc0, Hp(a1, b1))",
			# Then close the ring
			r1c="Sub(signAcc1, csum)",
			r1t="Sub(ri, Mul(r1c, s1))",
			# Ring consists of [(r0c, r0t), (r1c, r1t)]

			# First round of ring verify
			v0a="PointAdd(G(r0t), ScalarMult(p0, r0c))",				# g^t + y^c
			v0b="PointAdd(ScalarMult(m, r0t), ScalarMult(tau, r0c))",	# m^t + τ^c
			v0h="Hs(signAcc, Hp(v0a ,v0b))", 							# h = H(h, a, b)
			v0csum="r0c",												# csum += c

			# Second round of ring verify
			v1a="PointAdd(G(r1t), ScalarMult(p1, r1c))",				# g^t + y^c
			v1b="PointAdd(ScalarMult(m, r1t), ScalarMult(tau, r1c))",	# m^t + τ^c
			v1acc="Hs(v0h, Hp(v1a ,v1b))", 								# h = H(h, a, b)
			v1csum="r1c",												# csum += c

			# Ring verify result
			vcsum="Add(v0csum, v1csum)",
			verify="Equal(v1acc, vcsum)"
		))

	#a = T.term("Add(Mul(g, a), Mul(g, p))")
	#a = T.term("PointAdd(PointAdd(G(a), ScalarMult(G(p), t)), G(x))")
	#a = T.term("PointAdd(G(w), ScalarMult(G(p), s))")
	#a = T.term("Mul(Mul(x, y), Inv(y))")
	a = T.withvars(
		T.term("s"), dict(
			s="Add(d, Mul(k, r))",
			r="Add(Mul(k, x), Hs(m))",
		))
	a = T.term("ScalarMult(PointAdd(G(a), G(b)), c)")

	print("")
	print("Expr:", a)

	local_vars = extract_vars(a)
	result = eval_expr(str(a), local_vars)

	print("Vars:", ', '.join(local_vars.keys()))
	print("Result:", result)
	print("")
	for u in a.normalforms(rs):
		cmp_result = eval_expr(str(u), local_vars)
		print("=", u)
		if cmp_result != result:
			print("Error: expected:", result)
			print("            got:", cmp_result)

	print("")

	#T.show_tree(a, rs)

