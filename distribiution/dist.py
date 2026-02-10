
def get_distributions(df):    
    priority_dist = df["Priority"].value_counts()
    event_code_dist = df["EventCode"].value_counts()
    scenario_dist = df["Scenario"].value_counts()
    
    
    print("\nRozkład Priority")
    print(priority_dist)


    print("\nRozkład EventCode")
    print(event_code_dist)


    print("\nRozkład Scenario")
    print(scenario_dist)