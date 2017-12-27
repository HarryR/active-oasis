from __future__ import print_function
import sys
from oasis import rands, eval_expr
from collections import defaultdict


funcDefs = dict(
	# {returnType: [(funcName, [funcArgs], funcRet)]}
	p=[
		['G', ['s'], 'p'],
		['ScalarMult', ['p', 's'], 'p'],
		['PointNeg', ['p'], 'p'],
		['PointAdd', ['p', 'p'], 'p'],	
	],
	s=[
		['Sub', ['s', 's'], 's'],
		['Neg', ['s'], 's'],
		['Add', ['s', 's'], 's'],
		['Mul', ['s', 's'], 's'],
	]
)

args = dict(x=rands(), y=rands(), z=rands())


def genArgList(argTypeList, depth):
	for stuff in genArgs(argTypeList[0], depth):
		if len(argTypeList) > 1:
			for otherStuff in genArgList(argTypeList[1:], depth):
				yield ','.join([stuff, otherStuff])
		else:
			yield stuff


def genArgs(argType, depth=3):
	assert argType in funcDefs
	if argType == 's':
		for k in args.keys():
			yield k
	if argType == 's' and depth <= 0:
		return
	for funcName, funcArgTypes, funcRetType in funcDefs[argType]:
		if depth <= 0 and 'p' in funcArgTypes:
			continue
		for argType in genArgList(funcArgTypes, depth - 1):
			yield ''.join([funcName, '(', argType, ')'])


all_results = defaultdict(list)

for funcList in funcDefs.values():
	for funcName, funcArgTypes, funcRetType in funcList:
		for arg in genArgList(funcArgTypes, 2):
			expr = ''.join([funcName, '(', arg, ')'])
			result = eval_expr(expr, args)
			all_results[result].append(expr)
			print(expr, '=', result, file=sys.stderr)

for result, expressions in all_results.items():
	print("Result: ", result)
	for expr in expressions:
		print("\t", expr)
	print("")
