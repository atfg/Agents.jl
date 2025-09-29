## Required
const ABMPlot = Agents.get_ABMPlot_type()

Agents.agents_space_dimensionality(::Agents.HexSpaceSingle{D,Int}) where {D} = D

function Agents.get_axis_limits(model::ABM{<:Agents.HexSpaceSingle{2,Int}})
    e = axial_to_xy( oddr_to_axial( abmspace(model).extent) ) .+ 2
    o = zero.(e) .+ 0.5
    return o, e
end

function Agents.agentsplot!(ax::Axis, p::ABMP{<:HexSpaceSingle{2,Int}})
    hidedecorations!(ax)

    # Transform each odd-r point -> axial -> xy, then store as Point{2,Float32}
    # xy_coords = [
    #     let
    #         (row, col) = (Int(pt[1]), Int(pt[2]))            # interpret the float coords as (row, col)
    #         (q, r) = oddr_to_axial(row, col)            # convert to axial
    #         (x, y) = axial_to_xy(q, r)                  # axial -> xy
    #         Point{2,Float32}(Float32(x), Float32(y))        # final 2D point in Float32
    #     end for pt in p[:pos][]
    # ]

    # p[:pos][] = xy_coords

    if user_used_polygons(p.agent_marker[], p.marker[])
        poly!(p, p.marker; p.color, p.agentsplotkwargs...)
    else
        scatter!(p, p.pos; p.color, p.marker, p.markersize, p.agentsplotkwargs...)
    end
    return p
end

# function Agents.abmplot_pos(model::ABM{<:HexSpaceSingle{2,Int64}}, offset)
#     postype = Point2f
#     if isnothing(offset)
#         return postype[postype(model[i].pos) for i in allids(model)]
#     else
#         return postype[postype(model[i].pos .+ offset(model[i])) for i in allids(model)]
#     end
# end

"""
Plot the model's hex grid. Each cell is drawn as a scatter point with 
a hex marker. Occupied cells are colored differently from empty ones.
"""
function plot_hex_space(model::ABM{<:HexSpaceSingle{2,Int}}; marker_size=40)
    space = abmspace(model)
    w, h = space.extent[1], space.extent[2]

    # Gather all cell coords in bounding rectangle
    coords_oddr = [(q, r) for q in 1:w for r in 1:h]
    coords_axial = oddr_to_axial_tuple.(coords_oddr)
    
    # Convert each (q,r) to 2D for plotting
    xy = [axial_to_xy(c) for c in coords_axial]

    # Mark occupant or empty
    occupant_ids = [ get(space.occupancy, c, 0) for c in coords_oddr ]
    # Turn occupant ID 0 -> white, else any color
    colors = map(id -> id == 0 ? :white : model[id].color, occupant_ids)

    # We'll do a scatter with hex markers
    scatter(
        first.(xy),
        last.(xy),
        marker = :hex,
        markersize = 200/h,
        markercolor = colors,
        aspect_ratio = :equal,
        legend = false,
        xlims = (minimum(first.(xy)) - 1, maximum(first.(xy)) + 1),
        ylims = (minimum(last.(xy)) - 1, maximum(last.(xy)) + 1),
        title = "HexSpaceSingle Occupancy"
    )
end
