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

#blocksizes(::Type{QMPS_1Block{T}}, L, q, d) where {T} = [(q, d)]
nblocks(::Type{QMPS_1Block{T}}, L, q, d) where {T} = 1
blocksize(::Type{QMPS_1Block{T}}, L, q, d, i) where {T} = q, d

QMPS_1Block{T}(L, d, ig, a...) where {T} =
	createglobal(QMPS_1Block{T}, L, L-1, d, ig, a...)

QMPS_1Block{T}(L, q, d::Int, a...) where {T} = 
	(@assert q == L - 1; createglobal(QMPS_1Block{T}, L, q, d, a...))

