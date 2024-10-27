using Distributions
using LinearAlgebra
using ITensors

abstract type LocalCircuit end
const normaldist = Normal(0, 1)
# TODO: enable user to change \sigma
const weakdist = Normal(0, 0.0005)

struct LocalTensors
	tensors::Matrix{ITensor}
	i::Index
	q::Int
	d::Int
	ntensor::Int
end

function createlocal(::Type{T}, q, d, ii::Index, l::T, a...) where {T<:LocalCircuit}
	@assert q == l.ts.q && d > l.ts.d
	row, col = matsize(T, q, d, a...)
	tags = Matrix{NTuple{4, String}}(undef, row, col)
	tensors = Matrix{ITensor}(undef, row, col)
	ts = LocalTensors(tensors, ii, q, d, ntensors(T, q, d))
	obj = T(ts, process_args(T, a...)...)
	for i=1:row
		for j=1:col
			if isvalidind(T, obj, i, j, a...)
				tags[i, j] = gettag(T, obj, i, j, a...)
				t::Array{Float64, 4} = getarr(T, i, j, l)
				inds = (settags(ii, tg) for tg in tags[i, j])
				obj.ts.tensors[i, j] = ITensor(t, inds...)
			end
		end
	end
	return obj
end # Initialize from shallow Local ladder circuit with same number of qubits.

createlocal(::Type{T}) where T = T(nothing)
createlocal(::Type{T}, q, d, a...) where T = 
	createlocal(T, q, d, Index(2), a...)
createlocal(::Type{T}, q, d, ii::Index, a...) where T = 
	createlocal(T, q, d, ii, T(q), a...)
createlocal(::Type{T}, q, d, l::T, a...) where T = 
	createlocal(T, q, d, Index(2), l, a...)

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

# TODO: separate identity part to another function
function getarr(::Type{T}, i, j, l::T) where {T}
	if l.ts.d == 0
		return reshape(uniformso4(), 2, 2, 2, 2)
	elseif j <= l.ts.d
		t = l.ts.tensors[i, j].tensor
		noise_added = reshape(Vector(t.storage), 4, 4) * wnoise()
		return reshape(noise_added, 2, 2, 2, 2)
	else
		t = reshape(wnoise(), 2, 2, 2, 2)
		return permutedims(t, [1, 2, 4, 3])
	end
end
wnoise() = (A = rand(weakdist, 4, 4); exp((A - A')/2))

function uniformso4()
	vectors = [rand(normaldist, 4) for i=1:4]
	so4 = hcat(ortho(vectors...)...)
	if det(so4) < 0
		so4[:, 4] = -so4[:, 4]
	end
	return so4 
end

ortho() = Vector{Float64}[]
function ortho(v::Vector{Float64}, others::Vector{Float64}...)
	result_before::Vector{Vector{Float64}} = ortho(others...)
	for vb in result_before
		v = v - dot(v, vb) * vb
	end
	return [v / norm(v), result_before...]
end

include("LocalLadder.jl")
include("LocalBW.jl")
