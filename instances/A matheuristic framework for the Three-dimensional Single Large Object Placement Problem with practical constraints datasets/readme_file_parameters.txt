items parameters
	- client: client to be delivered
	- code: product code if necessary
	- height: box height
	- id: box id
	- lbH: load bearing in height direction
	- lbL: load bearing in lenght direction
	- lbW: load bearing in width direction
	- length: box length
	- material: box material
	- orientation: allowed orientations
	- priority: priority of item to be placed (lowest values -> higher priority)
	- product: box type
	- rotations: allowed rotation directions (Length, Width, Height)
	- weight: box weight
	- width: box width

bins parameters
	- amount: number of containers of this same type
	- code: code of container type
	- id: container id
	- height: container height
	- length: container length
	- width: container width
	- cost: cost of using container
	(load balance parameters)
	- horizontal distance between theoretical front-axle centre line and the back of the container (Hd)
	- minimum driving axle load: Tmin
	- maximum load transfer ratio: LTRmax
	- theoretical wheelbase: TWB
	- maximum rear axle load: RTmax
	- theoretical track width: T
	- horizontal distance between front-axle centre line and the centre of gravity: Hu
	- weight of unladen rear axle: RU
	- maximum payload: maximum weight limit
	- minimum steering axle load: Smin
	- maximum front axle load: FTmax
	- weight of unladen vehicle: U

practical constraints parameters
	- stacking constraints: if certain box types cannot be placed above others
	- loading priorities: if certain boxes have priority to be placed compared to others
	- multi-drop: if container(s) must stop in different spots
	- positioning: if certain boxes must be placed away from others
	- weight limit: if the maximum weight supported by the container must be respected
	- load balance: if boxes must be placed in a way that guarantee safety of transportation of the truck
	- load bearing: if boxes have a limit of weight per are that they support placed above them.

instance parameters
	- bin types: number of container types
	- item types: number of box types
	- clients: number of clients to deliver (multi-drop constraints)
	- name: instance name