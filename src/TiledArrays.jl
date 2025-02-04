module TiledArrays

using ChunkSplitters, LinearAlgebra

struct TiledArray{T, M<:AbstractArray{T}} <: AbstractArray{T, 2}
  tiles::Matrix{M}
  rowindices::Vector{UnitRange{Int64}}
  colindices::Vector{UnitRange{Int64}}
  isempties::Matrix{Bool}
  globalsize::Tuple{Int,Int}
  istransposed::Ref{Bool}
  function TiledArray(A::AbstractMatrix, rowindices::Vector{UnitRange{Int}}, colindices::Vector{UnitRange{Int}})
    B = Matrix{typeof(A)}(undef, length(rowindices), length(colindices))
    isempties = ones(Bool, length(rowindices), length(colindices))
    for (i, r) in enumerate(rowindices), (j, c) in enumerate(colindices)
      tile = view(A, r, c)
      iszero(tile) && continue
      isempties[i, j] = false
      B[i, j] = tile
    end
    return new{eltype(A), typeof(A)}(B, rowindices, colindices, isempties, size(A), Ref(false))
  end
end
function TiledArray(A::AbstractMatrix, ntiles::Integer)
  rowindices = collect(chunks(1:size(A, 1); n=ntiles))
  colindices = collect(chunks(1:size(A, 2); n=ntiles))
  return TiledArray(A, rowindices, colindices)
end
Base.size(B::TiledArray) = B.globalsize
Base.size(B::TiledArray, i) = 1 <= i <= 2 ? B.globalsize[i] : 1
Base.eltype(B::TiledArray{T}) where T = T
Base.length(B::TiledArray) = prod(size(B))
function LinearAlgebra.transpose!(B::TiledArray{T,M}) where {T,M}
  @assert length(B.rowindices) == length(B.colindices)
  B.istransposed[] = !B.istransposed[]
  tmp = B.rowindices
  B.rowindices .= B.colindices
  B.colindices .= tmp
  for i in eachindex(B.rowindices), j in eachindex(B.colindices)
    i > j && continue
    if !B.isempties[i, j] && !B.isempties[j, i]
      tmp = M(transpose(B.tiles[i, j]))
      B.tiles[i, j] = B.tiles[j, i]
      B.tiles[j, i] = tmp
    elseif B.isempties[i, j]
      B.tiles[i, j] = M(transpose(B.tiles[j, i]))
      B.tiles[j, i] = zeros(eltype(B), 0, 0)
    elseif B.isempties[j, i]
      B.tiles[j, i] = M(transpose(B.tiles[i, j]))
      B.tiles[i, j] = zeros(eltype(B), 0, 0)
    end
  end
  B.isempties .= B.isempties'
  return B
end
function Base.axes(B::TiledArray, i)
  i < 0 && throw(BoundsError("axes cannot be called with $i < 1"))
  1 <= i <= 2 && return Base.OneTo(size(B, i))
  return Base.OneTo(1)
end

tiles(B) = B.tiles
tiles(B, i, j) = B.tiles[i, j]
fastin(i::Integer, r::UnitRange) = r.start <= i <= r.stop
function rowindex(B::TiledArray, i::Int)
  itile = 0
  for (ii, r) in enumerate(B.rowindices)
    fastin(i, r) && (itile = ii; break)
  end
  return itile
end
function colindex(B::TiledArray, j::Int)
  jtile = 0
  for (jj, r) in enumerate(B.colindices)
    fastin(j, r) && (jtile = jj; break)
  end
  return jtile
end
function tileindices(B::TiledArray, i::Int, j::Int)
  return (rowindex(B, i), colindex(B, j))
end
function tileisempty(B::TiledArray, i::Int, j::Int)
  itile, jtile = tileindices(B, i, j)
  return B.isempties[itile, jtile]
end
function tilelocaldindices(B::TiledArray, i::Int, j::Int)
  itile, jtile = tileindices(B, i, j)
  return (i - B.rowindices[itile][1] + 1, j - B.colindices[jtile][1] + 1)
end

function Base.getindex(B::TiledArray, i::Int, j::Int)
  itile, jtile = tileindices(B, i, j)
  if B.isempties[itile, jtile]
    return zero(eltype(B))
  end
  li, lj = tilelocaldindices(B, i, j)
  return B.tiles[itile, jtile][li, lj]
end
Base.setindex!(B::TiledArray, v, ::Colon, js) = setindex!(B, v, 1:size(B, 1), js)
Base.setindex!(B::TiledArray, v, is, ::Colon) = setindex!(B, v, is, 1:size(B, 2))

function Base.setindex!(B::TiledArray{T, M}, v::Number, i::Integer, j::Integer) where {T,M}
#  iszero(v) && return v
  itile, jtile = first.(tileindices(B, i, j))
  if B.isempties[itile, jtile]
    B.isempties[itile, jtile] = false
    B.tiles[itile, jtile] = similar(M, (length(B.rowindices[itile]), length(B.colindices[jtile])))
  end
  li, lj = tilelocaldindices(B, i, j)
  B.tiles[itile, jtile][li, lj] = v
  flushzerotiles!(B)
  return v
end

function Base.setindex!(B::TiledArray{T}, v::AbstractArray{T}, is::AbstractVector{<:Integer}, js::AbstractVector{<:Integer}) where T
  for (ci, i) in enumerate(is), (cj, j) in enumerate(js)
    setindex!(B, v[ci, cj], i, j)
  end
  flushzerotiles!(B)
  return v
end
function flushzerotiles!(B::TiledArray)
  for (i, blck) in enumerate(B.tiles)
    if iszero(blck)
      B.isempties[i] = true
      B.tiles[i] = zeros(eltype(B), 0, 0)
    end
  end
  return B
end

export TiledArray

end # module TiledArrays
