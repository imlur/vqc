"""
locals : LocalCircuit objects. locals[i] is MPS for ith site
params : vector of 3-dim array, each dimensions means:
vector dim - at ith site
dim 1      - nth parameter of 
dim 2, 3   - (i, j) tensor
"""
struct QMPS_LBlocks{T<:LocalCircuit} <: QMPS{T}
	blk::QMPSBlocks{T}
end

