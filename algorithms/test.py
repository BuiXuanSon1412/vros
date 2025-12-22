def evaluate_with_timeline(self, solution: Solution) -> Dict:
    """
    Evaluate solution and return detailed timeline.
    """
    timeline = {"trucks": [[] for _ in range(self.data.number_of_trucks)], "drones": []}

    # Truck timeline
    truck_times = [0.0] * self.data.number_of_trucks

    for truck_idx, route in enumerate(solution.truck_routes):
        current_city = 0
        current_time = 0.0

        for node_idx, node in enumerate(route):
            if node_idx == 0:  # Depot
                timeline["trucks"][truck_idx].append(
                    {
                        "node": "depot",
                        "city_id": 0,
                        "arrival": 0.0,
                        "departure": 0.0,
                        "action": "start",
                    }
                )
                continue

            city = node[0]
            packages = node[1]

            # Travel time
            travel_time = self.data.manhattan_matrix[current_city][city]

            # Release date constraint
            max_release = 0
            if packages:
                max_release = max([self.data.release_date[p] for p in packages])

            arrival_time = max(current_time + travel_time, max_release)
            departure_time = arrival_time  # No service time for trucks

            timeline["trucks"][truck_idx].append(
                {
                    "node": f"city_{city}",
                    "city_id": city,
                    "arrival": arrival_time,
                    "departure": departure_time,
                    "waiting_time": max(0, max_release - (current_time + travel_time)),
                    "packages_picked": packages.copy(),
                }
            )

            current_time = departure_time
            current_city = city

        # Return to depot
        travel_time = self.data.manhattan_matrix[current_city][0]
        final_time = current_time + travel_time

        timeline["trucks"][truck_idx].append(
            {
                "node": "depot",
                "city_id": 0,
                "arrival": final_time,
                "departure": final_time,
                "action": "end",
            }
        )

        truck_times[truck_idx] = final_time

    # Drone timeline
    drone_available = [0.0] * self.data.number_of_drones

    for trip_idx, trip in enumerate(solution.drone_queue):
        drone_idx = np.argmin(drone_available)

        cities = [t[0] for t in trip]
        all_packages = []
        for t in trip:
            all_packages.extend(t[1])

        max_release = max([self.data.release_date[p] for p in all_packages])
        start_time = max(drone_available[drone_idx], max_release)

        trip_timeline = {
            "trip_id": trip_idx,
            "drone_id": drone_idx,
            "start_time": start_time,
            "stops": [],
        }

        current_time = start_time
        current_city = 0

        # Fly to each stop
        for stop_idx, node in enumerate(trip):
            city = node[0]
            packages = node[1]

            # Flight time
            flight_time = self.data.euclid_matrix[current_city][city]
            arrival = current_time + flight_time

            # Unloading
            departure = arrival + self.data.unloading_time

            trip_timeline["stops"].append(
                {
                    "city_id": city,
                    "arrival": arrival,
                    "departure": departure,
                    "packages_delivered": packages.copy(),
                }
            )

            current_time = departure
            current_city = city

        # Return to depot
        flight_time = self.data.euclid_matrix[current_city][0]
        end_time = current_time + flight_time

        trip_timeline["end_time"] = end_time
        timeline["drones"].append(trip_timeline)

        drone_available[drone_idx] = end_time

    return {
        "makespan": max(truck_times + list(drone_available)),
        "truck_times": truck_times,
        "timeline": timeline,
    }
