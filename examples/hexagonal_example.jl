using Agents, Random 
using CairoMakie
using Plots
using GLMakie

function random_color()
    rn = rand(3)
    return RGB(rn[1],rn[2],rn[3])
end 

@agent struct Cell(HexAgent{2})
    color::RGB{Float64} = random_color()
end

function clonecolor( agent )
    return(agent.color)
end

function markersize(model::ABM{<:HexSpaceSingle{2,Int}})
    return 550/abmspace(model).extent[1]
end

w = 9 # nr columns
h = 10 # nr rows

###############################################################################
# Example 1: move agent randomly (including staying in the same place)
###############################################################################

function agent_step_dance!(agent, model)
    remove_agent_from_space!(agent, model)
    rand_dir = random_axial_direction()
    
    q′ = agent.axial_pos[1] + rand_dir[1]
    r′ = agent.axial_pos[2] + rand_dir[2]
    
    axial_pos′ = (q′, r′)
    pos′ = axial_to_xy(axial_pos′)
    oddr_pos′ = axial_to_oddr(axial_pos′)

    space = abmspace(model)
    # Check bounding + occupancy
    if 1 <= oddr_pos′[1] <= space.extent[1] && 1 <= oddr_pos′[2] <= space.extent[2]
        if !haskey(space.occupancy, pos′)
            agent.pos = pos′
            agent.axial_pos = axial_pos′
        end
    end
    add_agent_to_space!(agent, model)
end

model = ABM(Cell, HexSpaceSingle{2,Int}(h,w); agent_step! = agent_step_dance!)
for i in 1:round(w*h*0.1)
    add_agent_single!(Cell, model)
end

abmvideo(
    "hexagonal_dance.mp4", model;
    agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon,
    framerate = 4, frames = 20,
    title = "The Dance of the Hexagons"
)

###############################################################################
# Example 2: travel along a row
###############################################################################

function agent_step_travel_column!(agent, model)
    remove_agent_from_space!(agent, model)
    oddr_pos = xy_to_oddr( agent.pos ) 
    oddr_pos′ = (oddr_pos[1], oddr_pos[2] + 1)
    axial_pos′ = oddr_to_axial(oddr_pos′)
    pos′ = oddr_to_xy(oddr_pos′)
    
    space = abmspace(model)
    # Check bounding + occupancy
    if 1 <= oddr_pos′[1] <= space.extent[1] && 1 <= oddr_pos′[2] <= space.extent[2]
        if !haskey(space.occupancy, pos′)
            agent.pos = pos′
            agent.axial_pos = axial_pos′
        end
    end
    add_agent_to_space!(agent, model)
end

model = ABM(Cell, HexSpaceSingle{2,Int}(h,w); agent_step! = agent_step_travel_column!)
for i in 1:round(w*h*0.1)
    add_agent_single!(Cell, model)
end

abmvideo(
    "/Users/filimon/Downloads/hexagonal_row_walk.mp4", model;
    agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon,
    framerate = 4, frames = 20,
    title = "The Dance of the Hexagons"
)

###############################################################################
# Example 3: expand
###############################################################################

function agent_step_expand!(agent, model)
    near_pos = nearby_positions(agent.pos, model)
    for cellpos in near_pos
        if isempty(cellpos, model)
            replicate!(agent, model; pos=cellpos )
        end
    end
end

model = ABM(Cell, HexSpaceSingle{2,Int}(h,w); agent_step! = agent_step_expand! )
add_agent_single!(Cell, model)

figure, _ = abmplot(model; agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon)
figure # returning the figure displays it

step!(model, 1)

figure, _ = abmplot(model; agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon)
figure # returning the figure displays it

step!(model, 1)

figure, _ = abmplot(model; agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon)
figure # returning the figure displays it


###############################################################################
# other tests
###############################################################################

model = ABM(Cell, HexSpaceSingle{2,Int}(h,w); agent_step! = agent_step_expand!)
add_agent_single!(Cell, model)

nearby_ids( model[1].pos, model )
collect( nearby_positions( model[1].pos, model ) )
randomwalk!(model[1], model; ifempty=true)

# create an agent and check the type and supertype
example_agent = Cell(id = 1, pos = (2, 3))
typeof(example_agent)
supertype(typeof(example_agent))

figure, ax, abmobs = abmplot(model; add_controls = true, agent_color = clonecolor, agent_size=markersize(model), agent_marker = :hexagon)
figure

###############################################################################
# Schelling example
###############################################################################

using Agents 

space = GridSpace((20, 20)) # 20×20 grid cells

@agent struct Schelling(GridAgent{2}) # inherit all properties of `GridAgent{2}`
    mood::Bool = false # all agents are sad by default :'(
    group::Int # the group does not have a default value!
end

function schelling_step!(agent, model)
  
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
  
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
  
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

properties = Dict(:min_to_be_happy => 3)

model = StandardABM(
    Schelling, # type of agents
    space; # space they live in
    agent_step! = schelling_step!, properties
)

for n in 1:300
    add_agent_single!(model; group = n < 300 / 2 ? 1 : 2)
end
