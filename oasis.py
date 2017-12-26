from __future__ import print_function
import sys
import trstools as T
from py_ecc import bn128
from random import randint
from hashlib import sha256
from py_ecc.bn128 import add, multiply, double, curve_order, field_modulus, G1, eq
from py_ecc.bn128.bn128_field_elements import inv


def bytes_to_int(x):
    o = 0
    for b in x:
        o = (o << 8) + ord(b)
    return o

rands = lambda: randint(1, curve_order - 1)
sbmul = lambda s: multiply(G1, s)
hashs = lambda *x: bytes_to_int(sha256('.'.join(['%X' for _ in range(0, len(x))]) % x).digest()) % curve_order
hashp = lambda *x: hashs(*[item.n for sublist in x for item in sublist])


expr_funcs = dict(	
	G=sbmul,
	Point=lambda x: x if isinstance(x, tuple) else sbmul(x),
	PointInfinity=lambda: None,
	PointDouble=lambda x: double(x),
	Neg=lambda x: -x % curve_order,
	Inv=lambda x: inv(x, curve_order),
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

	# Subtraction
	T.rule("Sub(Add(x, y), y) -> x"),
	T.rule("Sub(Add(x, y), x) -> y"),
	T.rule("Sub(Double(x), x) -> x"),

	# Multiplication
	T.rule("Mul(x, Mul(y, z)) -> Mul(Mul(x, y), z)"),
	T.rule("Mul(x, Half(y)) -> Half(Mul(x, y))"),
	T.rule("Mul(x, Double(y)) -> Double(Mul(x, y))"),

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
	#T.rule("PointAdd(Point(x), G(y)) -> PointAdd(Point(y), Point(x))"),
	#T.rule("PointAdd(Point(x), G(y)) -> PointAdd(Point(x), Point(y))"),
	#T.rule("PointAdd(Point(x), G(y)) -> PointAdd(Point(y), Point(x))"),
	#T.rule("PointAdd(Point(x), G(y)) -> PointAdd(Point(x), Point(y))"),

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
	T.rule("PointNeg(PointNeg(Point(x))) -> Point(x)"),
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

#for u,v in T.critical_pairs(rs):
#	print("[ %-16s -> %-12s ]" % (u, v))

#a = T.term("ScalarMult(G(x),y)")
#b = T.term("ScalarMult(G(y), x)")
#a = T.term("ScalarMult(G(x), y)")
#a = T.term("ScalarMult(ScalarMult(Point(x), y), z)")
#a = T.term("""ScalarMult(ScalarMult(PointAdd(Point(a), G(b)), c), z)""")
#a = T.term("Point(Add(w, Mul(p,z)))")
#a = T.term("ScalarMult(PointAdd(G(a), Point(b)), c)")
a = T.term("G(Add(Hs(c), Mul(Mul(Add(a, b),c), z)))")
#a = T.term("PointAdd(Point(Hs(c)),ScalarMult(Point(c),Mul(z,Add(b,a))))")
#a = T.term("Equal(Point(s),PointNeg(PointNeg(Point(s))))")
#a = T.term("Equal(Point(s), PointAdd(PointDouble(Point(s)),PointNeg(Point(s))) )")
#a = T.term("Equal(PointDouble(Point(a)),PointAdd(Point(a),Point(a)))")
#a = T.term("Equal(Neg(x),PointNeg(Point(x)))")

# Example of schnorr protocol
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

