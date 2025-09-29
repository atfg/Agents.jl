###############################################################################
# ToDo: 
# implement the torus
# what's the torus default?
# decide if using D everywhere or 2
# issues: randomwalk! doesn't work because my walk is not being called
# ids_in_position also doesn't work for interactive plots
###############################################################################

###############################################################################
# This file defines `HexSpaceSingle`. Notice that a lot of the space functionality
# comes from `AbstractGridSpace`, i.e., it is shared with `GridSpace`.
# This shared functionality is in the spaces/grid_general.jl file.
# The space also inherits a lot of discrete space functionality from spaces/discrete.jl.
###############################################################################

export HexSpaceSingle, HexAgent, random_position, add_agent_to_space!, remove_agent!
export remove_agent_from_space!, nearby_positions, nearby_ids, remove_all_from_space!
export oddr_to_axial, axial_to_oddr, random_axial_direction, xy_to_axial, axial_to_xy, xy_to_oddr, oddr_to_xy
#export offsets_within_radius

# Similar to `GridSpaceSingle` stored_ids is an array that directly
# stores IDs for each space position and uses ID=0 as an empty position.
# occupancy is a dictionary of xy coordinates
# stored_ids is in oddr coordinates
struct HexSpaceSingle{D,P} <: AbstractGridSpace{D,P}
    extent::NTuple{D,Int}
    occupancy::Dict{NTuple{D,P}, Int}
    stored_ids::Array{Int,D}
    metric::Function
    offsets_at_radius::Vector{Vector{NTuple{D,Int}}}
    offsets_within_radius::Vector{Vector{NTuple{D,Int}}}
    offsets_within_radius_no_0::Vector{Vector{NTuple{D,Int}}}
end

function Base.size(space::HexSpaceSingle{2,Int})
    return (space.extent)
end

"""
    HexSpaceSingle( width, height )

A hexagonal 2D grid space of size `width` × `height`. This space allows only one
agent per position. `occupancy` is a Dict that maps grid coordinates to the 
list of agent IDs currently occupying that position.
"""
function HexSpaceSingle{D,P}(width::P, height::P) where {D,P}
    # Throw an error if a user tries to instantiate with D ≠ 2.
    if D != 2
        throw(ArgumentError("HexSpaceSingle must be 2D, but got D = $D."))
    end

    # Define a default distance function, e.g. hex distance in axial coords:
    distfun(a::NTuple{2,P}, b::NTuple{2,P}) = hex_distance(a, b)

    s = zeros(Int, (width,height))

    return HexSpaceSingle{2,P}(
        (width,height),
        Dict{NTuple{2,P}, Int}(),  # occupancy starts empty
        s,                         # stored_ids, matrix of 0s
        distfun,
        Vector{Vector{NTuple{D,Int}}}(),
        Vector{Vector{NTuple{D,Int}}}(),
        Vector{Vector{NTuple{D,Int}}}()
    )
end

###############################################################################
# Helper functions to convert between xy, offset and axial coordinates
###############################################################################

"""
    axial_to_oddr(axial::NTuple{2,Int}) -> (row, col)

Convert from axial coordinates `(q, r)` to odd-r offset coordinates `(row, col)`.
Equivalent to the old `axial_to_oddr(q::Int, r::Int)`.
"""
function axial_to_oddr(axial::NTuple{2,Int})
    q, r = axial
    col = q + (r - (r & 1)) ÷ 2
    row = r
    return (row, col)
end

"""
    oddr_to_axial(oddr::NTuple{2,Int}) -> (q, r)

Convert from odd-r offset `(row, col)` to axial `(q, r)`.
Equivalent to the old `oddr_to_axial(row::Int, col::Int)`.
"""
function oddr_to_axial(oddr::NTuple{2,Int})
    row, col = oddr
    q = col - (row - (row & 1)) ÷ 2
    r = row
    return (q, r)
end

"""
    oddr_to_xy(oddr::NTuple{2,Int}) -> (x, y)

Convert from odd-r offset `(row, col)` to integer `(x, y)` coordinates, 
using 2*col plus a +1 shift for odd rows. 
Equivalent to the old `oddr_to_xy(row::Int, col::Int)`.
"""
function oddr_to_xy(oddr::NTuple{2,Int})
    row, col = oddr
    x = 2*col + (row & 1)
    y = row
    return (x, y)
end

"""
    xy_to_oddr(xy::NTuple{2,Int}) -> (row, col)

Inverse of `oddr_to_xy`. Convert integer `(x, y)` in our odd-r layout 
to `(row, col)`. 
Equivalent to the old `xy_to_oddr(x::Int, y::Int)`.
"""
function xy_to_oddr(xy::NTuple{2,Int})
    x, y = xy
    row = y
    if iseven(row)
        col = x ÷ 2
    else
        col = (x - 1) ÷ 2
    end
    return (row, col)
end

"""
    axial_to_xy(axial::NTuple{2,Int}) -> (x, y)

Convert from axial `(q, r)` directly to `(x, y)` by 
    (q, r) --axial_to_oddr--> (row, col) --oddr_to_xy--> (x, y).
Equivalent to the old `axial_to_xy(q::Int, r::Int)`.
"""
function axial_to_xy(axial::NTuple{2,Int})
    # destruct axial coords
    row_col = axial_to_oddr(axial)
    return oddr_to_xy(row_col)
end

"""
    xy_to_axial(xy::NTuple{2,Int}) -> (q, r)

Inverse of `axial_to_xy`. Convert `(x, y)` to `(q, r)` by
    (x, y) --xy_to_oddr--> (row, col) --oddr_to_axial--> (q, r).
Equivalent to the old `xy_to_axial(x::Int, y::Int)`.
"""
function xy_to_axial(xy::NTuple{2,Int})
    row_col = xy_to_oddr(xy)
    return oddr_to_axial(row_col)
end



# """
#     axial_to_oddr(q, r) -> (row, col)

# Convert from axial (q, r) coords to "odd-r" offset coordinates (row, col).
# In "odd-r", every odd row is shifted right by half a cell.

# Returns a tuple (row, col) of integers.
# """
# function axial_to_oddr(q::Int,r::Int)
#     col = q + (r - (r & 1)) ÷ 2
#     row = r
#     return (row, col)
# end

# function axial_to_oddr(q::Int,r::Int)
#     col = q + (r - (r & 1)) ÷ 2
#     row = r
#     return (row, col)
# end
# pos::NTuple{2,Int}

# """
#     oddr_to_axial(row, col) -> (q, r)

# Convert from "odd-r" offset (row, col) coords back to axial (q, r).
# Assumes the row, col correspond to the "odd-r" horizontal layout.

# Returns a tuple (q, r) of integers.
# """
# function oddr_to_axial(row::Int, col::Int)
#     q = col - (row - (row & 1)) ÷ 2
#     r = row
#     return (q,r)
# end

# """
#     oddr_to_xy(row, col) -> (x, y)

# Map "odd-r offset" coordinates (row, col) to a simple integer (x,y) grid.
# We shift every odd row by +1 in the x-direction, ensuring that neighbors
# line up in a hex-adjacency pattern without using √3.
# """
# function oddr_to_xy(row::Int, col::Int)
#     # If row is even, x = 2*col + 0
#     # If row is odd,  x = 2*col + 1
#     x = 2*col + (row & 1)
#     y = row
#     return (x, y)
# end

# """
#     xy_to_oddr(x::Int, y::Int) -> (row, col)

# Inverse of oddr_to_xy. Given integer (x, y) in our "integer odd-r" layout,
# return the (row, col) in odd-r offset coordinates.

# Assumes:
# - `y` is the row.
# - For even `row`, `x` = 2*col.
# - For odd  `row`, `x` = 2*col + 1.
# """
# function xy_to_oddr(x::Int, y::Int)
#     row = y
#     if iseven(row)
#         col = x ÷ 2
#     else
#         col = (x - 1) ÷ 2
#     end
#     return (row, col)
# end

# """
#     axial_to_xy(q, r) -> (x, y)

# Convert from axial (q, r) to integer coordinates (x, y) in an "odd-r" layout.
# Internally does:
#     (q, r) --axial_to_oddr--> (row, col) --oddr_to_xy--> (x, y)
# """
# function axial_to_xy(q::Int, r::Int)
#     row, col = axial_to_oddr(q, r)
#     x, y = oddr_to_xy(row, col)
#     return (x, y)
# end

# """
#     xy_to_axial(x, y) -> (q, r)

# Inverse of `axial_to_xy`. 
# Converts integer coords (x, y) in the odd-r layout back to (q, r) in axial:
#     (x, y) --xy_to_oddr--> (row, col) --oddr_to_axial--> (q, r)
# """
# function xy_to_axial(x::Int, y::Int)
#     row, col = xy_to_oddr(x, y)
#     q, r = oddr_to_axial(row, col)
#     return (q, r)
# end

###############################################################################
# Minimal Agent Type
###############################################################################

# An example agent to be used in this HexSpaceSingle. We store its position as 
# xy converted offset coordinates (row, col), while axial_pos stores axial 
# coordinates (q, r).
"""
    HexAgent <: AbstractAgent
The minimal agent struct for usage with [`HexSpaceSingle`](@ref).
It stores position in both oddr offset (`pos`) and axial coordinates (`axial_pos`).
"""
@agent struct HexAgent{D}(NoSpaceAgent)
    pos::NTuple{D, Int}
    #axial_pos::Tuple{Int, Int} = oddr_to_axial( pos[1], pos[2] )
    axial_pos::Tuple{Int, Int} = xy_to_axial( pos )
end

###############################################################################
# Implementation of the Space API
###############################################################################

"""
    random_position(model)

Return a random valid position in xy coordinates in the hexagonal grid of `model.space`.
"""
function random_position(model::ABM{<:HexSpaceSingle{2,Int}})
    # first generate oddr coordinates
    row = rand(1:spacesize(model)[1])
    col = rand(1:spacesize(model)[2])
    # then convert to xy
    return oddr_to_xy( (row, col) )
end

"""
    add_agent_to_space!(agent, model)

Place `agent` in the occupancy list for its position `agent.pos`.
"""
function add_agent_to_space!(agent::AbstractAgent, model::ABM{<:HexSpaceSingle}) 
    space = abmspace(model)
    pos = agent.pos
    
    # Ensure there's a slot to store agent IDs for this position
    if haskey(space.occupancy, pos)
        error("Cell is already occupied by agent ID $(space.occupancy[pos]).")
    end

    if !haskey(space.occupancy, pos)
        space.occupancy[pos] = agent.id
    end
    
    pos′ = xy_to_oddr( pos )
    abmspace(model).stored_ids[pos′...] = agent.id
end

"""
    remove_agent_from_space!(agent, model)

Remove `agent` from the occupancy list of its position.
"""
function remove_agent_from_space!(agent::AbstractAgent, model::ABM{<:HexSpaceSingle{2,Int}})
    space = abmspace(model)
    if get(space.occupancy, agent.pos, nothing) == agent.id
        delete!(space.occupancy, agent.pos)
        pos′ = xy_to_oddr( agent.pos )
        abmspace(model).stored_ids[pos′...] = 0
    end
end

"""
    nearby_ids(pos, model, r)

Return a list of agent IDs that lie within hex distance `r` of the hex coordinate
`pos`. 
"""
function nearby_ids(pos::NTuple{2,Int}, model::ABM{<:HexSpaceSingle{2,Int}}, r::Int)
    #axial_pos = oddr_to_axial(pos[1],pos[2])
    axial_pos = xy_to_axial(pos)
    space = abmspace(model)
    (w, h) = space.extent
    results = Int[]

    for dq in -r:r
        for dr in -r:r
            q′ = axial_pos[1] + dq
            r′ = axial_pos[2] + dr

            axial_pos′ = (q′, r′)
            oddr_pos′ = axial_to_oddr( axial_pos′ )
            pos′ = axial_to_xy( axial_pos′ )

            # skip out-of-bounds (no wrap)
            if oddr_pos′[1] < 1 || oddr_pos′[1] > h || oddr_pos′[2] < 1 || oddr_pos′[2] > w
                continue
            end

            # check hex distance
            if hex_distance(axial_pos, axial_pos′) <= r
                occupant = get(space.occupancy, pos′, nothing)
                if occupant !== nothing
                    push!(results, occupant)
                end
            end
        end
    end
    return results
end

# `random_empty` comes from spaces/discrete.jl as long as we extend:
function Base.isempty(pos::NTuple{D,Int}, model::ABM{<:HexSpaceSingle{2,Int}}) where {D}
    pos′ = xy_to_oddr( pos )
    return abmspace(model).stored_ids[pos′...] == 0
end

# And we also need to extend the iterator of empty positions
# positions(model) gives cartesian indices which are in oddr
function empty_positions(model::ABM{<:HexSpaceSingle{2,Int}})
    Iterators.filter(i -> abmspace(model).stored_ids[i...] == 0, positions(model))
end

"""
    id_in_position(pos, model) → id

Return the agent ID in the given position.
This will be `0` if there is no agent in this position.
This is similar to [`ids_in_position`](@ref), but specialized for `HexSpaceSingle`.
See also [`isempty`](@ref).
"""
function id_in_position(pos::NTuple{D,Int}, model::ABM{<:HexSpaceSingle{D,Int}}) where {D}
    return abmspace(model).stored_ids[xy_to_oddr(pos)...]
end

###############################################################################
# Helper functions: 
#   hex distance in axial coordinates
#   random axial directions
#   convert axial coords to approximate 2D coords for plotting
###############################################################################
"""
    hex_distance(a, b)

Compute distance between two axial-coord points (q,r) using standard
axial-coord formula. Works for integer P.
"""
function hex_distance(a::NTuple{2,T}, b::NTuple{2,T}) where {T<:Real}
    x1, z1 = a
    x2, z2 = b
    y1 = -x1 - z1
    y2 = -x2 - z2
    return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) ÷ 2
end

function random_axial_direction()
    directions = calculate_offsets(1)
    return rand(directions)
end

function calculate_offsets(N::Int)
    results = NTuple{2,Int}[]
    for q in -N:N
        for r in max(-N, -q - N):min(N, -q + N)
            push!(results, (q, r))
        end
    end
    return results
end

#######################################################################################
# Implementation of nearby_stuff
#######################################################################################
# The following functions utilize the 1-agent-per-position knowledge,
# hence giving faster nearby looping than `GridSpace`.
# Notice that the code here is a near duplication of `nearby_positions`
# defined in spaces/grid_general.jl. Unfortunately
# the duplication is necessary because `nearby_ids(pos, ...)` should in principle
# contain the id at the given `pos` as well.

"""
    offsets_within_radius(space::HexSpaceSingle{D,Int}, r::Real)
The function does two things:
1. If a vector of indices exists in the model, it returns that.
2. If not, it creates this vector, stores it in the model and then returns that.
"""
function offsets_within_radius(space::HexSpaceSingle{D,Int}, r::Real) where {D}
    i = floor(Int, r + 1)
    offsets = space.offsets_within_radius
    if isassigned(offsets, i)
        βs = offsets[i]
    else
        r₀ = i - 1
        βs = calculate_offsets(r₀)
        resize_offsets!(offsets, i)
        offsets[i] = βs
    end
    return βs
end

function offsets_within_radius_no_0(space::HexSpaceSingle{D,Int}, r::Real) where {D}
    i = floor(Int, r + 1)
    offsets = space.offsets_within_radius_no_0
    if isassigned(offsets, i)
        βs = offsets[i]
    else
        r₀ = i - 1
        βs = calculate_offsets(r₀)
        z = ntuple(i -> 0, D)
        filter!(x -> x ≠ z, βs)
        resize_offsets!(offsets, i)
        offsets[i] = βs
    end
    return βs
end

"""
    offsets_at_radius(space::HexSpaceSingle{D,Int}, r::Real)
The function does two things:
1. If a vector of indices exists in the model, it returns that.
2. If not, it creates this vector, stores it in the model and then returns that.
Offset are for axial coordinates
"""
function offsets_at_radius(space::HexSpaceSingle{D,Int}, r::Real) where {D}
    i = floor(Int, r + 1)
    offsets = space.offsets_at_radius
    if isassigned(offsets, i)
        βs = offsets[i]
    else
        r₀ = i - 1
        βs = calculate_offsets(r₀)

        #println( "Hello!!!!" )
        #println( βs )
        filter!(β -> space.metric(β,(0,0)) == r₀, βs)
        #println( βs )
        
        resize_offsets!(offsets, i)
        offsets[i] = βs
    end
    return βs
end

# without torus
function nearby_ids(pos::NTuple{D,Int}, model::ABM{<:HexSpaceSingle{2,Int}}, r=1,
    get_offset_indices=offsets_within_radius # internal, see last function
) where {D}
    nindices = get_offset_indices(model, r)
    stored_ids = abmspace(model).stored_ids
    space_size = spacesize(model)
    axial_pos = xy_to_axial(pos)
    oddr_pos = axial_to_oddr( axial_pos )
    # in oddr coordinates
    position_iterator = (axial_to_oddr(axial_pos .+ β) for β in nindices)
   
    # check if we are far from the wall to skip bounds checks
    if oddr_pos[1] > r && oddr_pos[1] <= space_size[1] - r && oddr_pos[2] > r && oddr_pos[2] <= space_size[2] - r
        ids_iterator = (stored_ids[p...] for p in position_iterator
                        if stored_ids[p...] != 0)
    else # do boundary checks
        ids_iterator = (stored_ids[p...] for p in position_iterator
                        if checkbounds(Bool, stored_ids, p...) && stored_ids[p...] != 0)
    end
    return ids_iterator
end

"""
    nearby_positions(pos, model::ABM{<:DiscreteSpace}, r=1; kwargs...)

Return an iterable of all positions within "radius" `r` of the given `position`
(which excludes given `position`).
The `position` must match type with the spatial structure of the `model`.

The value of `r` and possible keywords operate identically to [`nearby_ids`](@ref).

This function only exists for discrete spaces with a finite amount of positions.

    nearby_positions(position, model::ABM{<:OpenStreetMapSpace}; kwargs...) → positions

"""
function nearby_positions(pos::NTuple{D,Int}, model::ABM{<:HexSpaceSingle{2,Int}}, r=1,
    get_offset_indices=offsets_within_radius_no_0
) where {D}
    nindices = get_offset_indices(model, r)
    space_size = spacesize(model)
    stored_ids = abmspace(model).stored_ids
    axial_pos = xy_to_axial(pos)
    oddr_pos = axial_to_oddr(axial_pos)
    # in oddr coordinates
    oddr_position_iterator = (axial_to_oddr(axial_pos .+ β) for β in nindices)
    # in xy coordinates
    position_iterator = (axial_to_xy(axial_pos .+ β) for β in nindices)

    # check if we are far from the wall to skip bounds checks
    if oddr_pos[1] > r && oddr_pos[1] <= space_size[1] - r && oddr_pos[2] > r && oddr_pos[2] <= space_size[2] - r
        return position_iterator
    else # do boundary checks
        return (oddr_to_xy(p) for p in oddr_position_iterator if checkbounds(Bool, stored_ids, p...) )
    end
end

function walk!(
    agent::AbstractAgent,
    direction::NTuple{D,Int},
    model::ABM{<:HexSpaceSingle{2,Int}}
) where {D}
    println("IM HEREEEEEE!")
    target = xy_to_axial(pos) .+ direction

    target = oddr_to_xy( normalize_position( xy_to_oddr(target), model) )
    if isempty(target, model) # if target unoccupied
        move_agent!(agent, target, model)
    end
    return agent
end