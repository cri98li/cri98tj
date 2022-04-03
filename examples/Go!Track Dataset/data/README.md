# GPS Trajectories Data Set

https://archive.ics.uci.edu/ml/datasets/GPS+Trajectories


### go_track_tracks.csv: a list of trajectories
- **id_android**: it represents the device used to capture the instance;
- **speed**: it represents the average speed (Km/H)
- **distance**: it represent the total distance (Km)
- **rating**: it is an evaluation parameter. Evaluation the traffic is a way to verify the volunteers perception about the traffic during the travel, in other words, if volunteers move to some place and face traffic jam, maybe they will evaluate 'bad'. (3- good, 2- normal, 1-bad).
- **rating_bus**: it is other evaluation parameter. (1 - The amount of people inside the bus is little, 2 - The bus is not crowded, 3- The bus is crowded.
- **rating_weather**: it is another evaluation parameter. ( 2- sunny, 1- raining).
- **car_or_bus**: (1 - car, 2-bus)
- **linha**: information about the bus that does the pathway



### go_track_trackspoints.csv: localization points of each trajectory
- **id**: unique key to identify each point
- **latitude**: latitude from where the point is
- **longitude**: longitude from where the point is
- **track_id**: identify the trajectory which the point belong
- **time**: datetime when the point was collected (GMT-3)

Missing values: YES