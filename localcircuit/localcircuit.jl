using LinearAlgebra
using ITensors

abstract type LocalCircuit end

struct LocalTensors
	tensors::Matrix{ITensor}
	i::Index
	q::Int
	d::Int
	ntensor::Int
end

function buildtensors(::Type{T}, q, d, ii, l::T, a...) where {T<:LocalCircuit}
	@assert q == l.ts.q && d > l.ts.d
	row, col = matsize(T, q, d, a...)
	tags = Matrix{NTuple{4, String}}(undef, row, col)
	tensors = Matrix{ITensor}(undef, row, col)
	for i=1:row
		for j=1:col
			tags[i, j] = gettag(T, q, d, i, j, l, a...)
			t::Array{Float64, 4} = getarr(T, q, d, i, j, l)
			inds = (settags(ii, tg) for tg in tags[i, j])
			tensors[i, j] = ITensor(t, inds...)
		end
	end
	return tensors
end # Initialize from shallow Local ladder circuit with same number of qubits.

createlocal(::Type{T}) where T = T(nothing)
createlocal(::Type{T}, q, d, a...) where T = 
	createlocal(T, q, d, Index(2), a...)
createlocal(::Type{T}, q, d, ii::Index, a...) where T = 
	createlocal(T, q, d, ii, T(q), a...)
createlocal(::Type{T}, q, d, l::T, a...) where T = 
	createlocal(T, q, d, Index(2), l, a...)

function createlocal(::Type{T}, q, d, ii::Index, l::T, a...) where {T<:LocalCircuit}
	ts = LocalTensors(buildtensors(T, q, d, ii, l, a...),
		ii, q, d, ntensors(T, q, d))
	other_args = process_args(T, a...)
	return T(ts, other_args...)
end

nth_tensor(l::T, i) where {T<:LocalCircuit} = 
	l.ts.tensors[nth_tensor_idx(T, l, i)...]

contract(l, tinit=ITensor(1)::ITensor) = 
	contract(l, 1, l.ts.ntensor, tinit)
contract(l, e::Int, tinit=ITensor(1)::ITensor) = 
	contract(l, 1, e, tinit)
rcontract(l, tinit=ITensor(1)::ITensor) = 
	contract(l, l.ts.ntensor, 1, tinit)
rcontract(l, e::Int, tinit=ITensor(1)::ITensor) = 
	contract(l, l.ts.ntensor, e, tinit)

function contract(l, s, e, t)
	@assert 1 <= s && e <= l.ts.ntensor
	range = s <= e ? (s:e) : (s:-1:e)	
	for i=range
		t *= nth_tensor(l, i)
	end
	return t
end

# TODO: change the code to "Uniform" SO(4) (4 4D vectors and orthonormalize)
# TODO: give small noise to resultant matrix
# TODO: separate identity part to another function
function getarr(::Type{T}, q, d, i, j, l::T) where {T}
	if l.ts.d == 0
		A = rand(4, 4)
		return reshape(exp((A - A')/2), 2, 2, 2, 2)
	elseif j <= l.ts.d
		t = l.ts.tensors[i, j].tensor
		return reshape(Vector(t.storage), 2, 2, 2, 2)
	else
		t = reshape(Matrix(I, 4, 4), 2, 2, 2, 2)
		return permutedims(t, [1, 2, 4, 3])
	end
end

include("LocalLadder.jl")
include("LocalBW.jl")
