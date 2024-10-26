"""
locals : LocalCircuit objects. locals[i] is MPS for ith site
params : vector of 3-dim array, each dimensions means:
vector dim - at ith site
dim 1      - nth parameter of 
dim 2, 3   - (i, j) tensor
"""
struct QMPS_1Block{T<:LocalCircuit} <: QMPS{T}
	blk::QMPSBlocks{T}
end

QMPS_1Block{T}(L, q, d) where {T} = QMPS_1Block{T}(L, q, d, QMPS_1Block{T}())

# TODO: add initialization code similar to localcircuit structs
function QMPS_1Block{T}(L, q, d, g::QMPS_1Block{T}) where {T}
	@assert L == g.ts.L && q == g.ts.q && d > g.ts.d
	locals = Vector{T}(undef, L)
	params = get_params(locals)
	QMPS_1Block{T}(locals, params, L, q, d)
end
