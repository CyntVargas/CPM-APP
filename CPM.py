import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def convert_predecessors(x):
    """
    Convert 'Predecessors' column to list of integers.
    """
    if pd.isna(x):
        return
    elif isinstance(x, (int, float)):
        return [int(x)]
    elif isinstance(x, str):
        return [int(pred.strip()) for pred in x.split(',')]
    else:
        return

def all_paths_with_duration(df):
    """
    Finds all sequential paths through a set of activities and their predecessors,
    handling multiple predecessors for each activity, and calculates the duration of each path.

    Args:
        df (pd.DataFrame): DataFrame with 'Activity' and 'Predecessors' columns.

    Returns:
        dict: A dictionary where keys are activities and values are lists of dictionaries,
              each containing the path and its total duration.
    """

    activity_map = df.set_index('Activity').to_dict('index')
    paths_with_durations = {}

    for start_activity in activity_map:
        paths_with_durations[start_activity] =[]
        queue = [[[start_activity], start_activity,
                  activity_map[start_activity]['Duration']]]  # [path, current_activity, current_duration]

        while queue:
            current_path, current_activity, current_duration = queue.pop(0)
            next_activities =[]
            for act, data in activity_map.items():
                predecessors = data.get('Predecessors', )
                if predecessors is None:
                    predecessors =[]
                if current_activity in predecessors:
                    next_activities.append(act)

            if not next_activities:
                paths_with_durations[start_activity].append({
                    'path': current_path,
                    'duration': current_duration
                })
            else:
                for next_activity in next_activities:
                    next_duration = current_duration + activity_map[next_activity]['Duration']
                    new_path = current_path + [next_activity]
                    queue.append([new_path, next_activity, next_duration])

    return paths_with_durations

def create_network_diagram(df, x_coordinate_dict, y_coordinate_dict, radius, x_offset, y_offset,
                               project_start_date=None, critical_path_duration=None, all_paths_with_durations=None):
    """
    Creates a network diagram using Plotly with arrows.

    Args:
        df (pd.DataFrame): DataFrame containing the activity numbers.
        x_coordinate_dict (dict): A dictionary containing the x-coordinates for each activity.
        y_coordinate_dict (dict): A dictionary containing the y-coordinates for each activity.
        radius (float): The radius of the circles.
        x_offset (float): Offset for the text along the x-axis.
        y_offset (float): Offset for the text along the y-axis.
        project_start_date (str, optional): The project's start date in 'YYYY-MM-DD' format.
                                           If provided, the x-axis will show dates.
        critical_path_duration (int, optional): The duration of the critical path.
        all_paths_with_durations (dict, optional):  Dictionary containing all paths and their durations.
    """
    total_duration = df['Duration'].sum()

    # Create initial figure
    fig_initial = go.Figure()

    # Convert x-coordinates to datetime if project_start_date is provided
    if project_start_date:
        start_date = datetime.strptime(project_start_date, '%Y-%m-%d')
        x_coordinate_dict = {
            activity: start_date + timedelta(days=x)
            for activity, x in x_coordinate_dict.items()
        }

    # Draw circles and text
    for i in range(len(df)):
        activity = df["Activity"].iloc[i]
        x_coordinate = x_coordinate_dict[activity]
        y_coordinate = y_coordinate_dict[activity]

        # Calculate offsets in terms of days (timedelta) if using dates
        if project_start_date:
            radius_offset = timedelta(days=radius / 3.5)
            text_x_offset = timedelta(days=x_offset)

        else:
            radius_offset = radius / 3.5
            text_x_offset = x_offset

        # Circle
        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate],
            y=[y_coordinate],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=radius * 20,
                color="white",
                line=dict(color="black", width=1)
            ),
            hoverinfo="skip"
        ))

        # Horizontal line inside the circle
        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate - radius_offset, x_coordinate + radius_offset],
            y=[y_coordinate, y_coordinate],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip"
        ))

        # Vertical line inside the circle
        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate, x_coordinate],
            y=[y_coordinate - radius / 2, y_coordinate],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip"
        ))

        # Add activity number inside the circle with offset
        duration = df["Duration"].iloc[i]

        # Activity text
        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate],
            y=[y_coordinate + y_offset],
            mode="text",
            text=[str(activity)],
            textposition="middle center",
            hoverinfo="skip"
        ))

        # Days since start
        if project_start_date:
            x_coordinate_duration = x_coordinate - text_x_offset
            x_coordinate_finish = x_coordinate + text_x_offset
        else:
            x_coordinate_duration = x_coordinate - text_x_offset
            x_coordinate_finish = x_coordinate + text_x_offset
        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate_duration],
            y=[y_coordinate - y_offset],
            mode="text",
            text=[str(x_coordinate)],
            textposition="middle center",
            hoverinfo="skip"
        ))

        # Time to finish text
        time_to_finish = calculate_time_to_finish(df, activity, x_coordinate_dict, y_coordinate_dict,
                                                  all_paths_with_durations, critical_path_duration)

        fig_initial.add_trace(go.Scatter(
            x=[x_coordinate_finish],
            y=[y_coordinate - y_offset],
            mode="text",
            text=[str(time_to_finish)],
            textposition="middle center",
            hoverinfo="skip"
        ))

    # Draw connections with arrows
    activity_map = df.set_index('Activity').to_dict('index')
    branching_points = {}
    for act, data in activity_map.items():
        predecessors = data.get('Predecessors')
        if predecessors:
            for predecessor in predecessors:
                if predecessor not in branching_points:
                    branching_points[predecessor] = 0
                branching_points[predecessor] += 1

    for activity, data in activity_map.items():
        x_end = x_coordinate_dict[activity]  # Successor's x
        y_end = y_coordinate_dict[activity]  # Successor's y
        predecessors = data.get('Predecessors')
        if predecessors:
            for predecessor in predecessors:
                x_start = x_coordinate_dict[predecessor]  # Predecessor's x
                y_start = y_coordinate_dict[predecessor]  # Predecessor's y

                # Determine line color based on y-coordinates
                if y_coordinate_dict[activity] == 0 and y_coordinate_dict[predecessor] == 0:
                    line_color = "red"
                    arrow_color = "red"
                else:
                    line_color = "black"
                    arrow_color = "black"

                # Check if the predecessor is a branching point
                is_branching_point = branching_points.get(predecessor, 0) > 1

                if is_branching_point:
                    # Draw vertical line first
                    fig_initial.add_trace(go.Scatter(
                        x=[x_start, x_start],
                        y=[y_start, y_end],
                        mode="lines",
                        line=dict(color=line_color, width=1),
                        hoverinfo="skip"
                    ))
                    # Draw horizontal line
                    fig_initial.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[y_end, y_end],
                        mode="lines",
                        line=dict(color=line_color, width=2),
                        hoverinfo="skip"
                    ))
                    # Add arrowhead (marker) at the end of the horizontal line
                    fig_initial.add_trace(go.Scatter(
                        x=[x_end],
                        y=[y_end],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-right",
                            size=8,
                            color=arrow_color
                        ),
                        hoverinfo="skip"
                    ))
                else:
                    # Draw direct line
                    fig_initial.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[y_start, y_start],
                        mode="lines",
                        line=dict(color=line_color, width=2),
                        hoverinfo="skip"
                    ))
                    # Add arrowhead
                    fig_initial.add_trace(go.Scatter(
                        x=[x_end],
                        y=[y_start],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-right",
                            size=8,
                            color=arrow_color
                        ),
                        hoverinfo="skip"
                    ))

    # Create a copy for correction
    fig_corrected = go.Figure(fig_initial)

    # --- Correction Loop ---
    for trace in fig_corrected.data:
        if trace.mode == "lines" and trace.line.color in ("red", "black"):  # Check if it's a connection line
            if len(trace.y) == 2 and trace.y[0] == 0 and trace.y[1] == 0:
                # Find the activity connected to
                connected_activity = None
                for activity, coords in y_coordinate_dict.items():
                    if y_coordinate_dict[activity] == trace.y[0] and x_coordinate_dict[activity] == trace.x[1]:
                        connected_activity = activity
                        break
                if connected_activity is not None:
                    trace.y = [y_coordinate_dict[connected_activity], y_coordinate_dict[connected_activity]]

    # Add a dummy trace for the visible x-axis
    x_min = min(x_coordinate_dict.values())
    x_max = max(x_coordinate_dict.values())

    dummy_x_data = [x_min, x_max]

    # If project_start_date is provided, convert x-coordinates to dates
    if project_start_date:
        start_date = datetime.strptime(project_start_date, '%Y-%m-%d')
        dummy_x_data = [x_min, x_max]
        dummy_x_data = [min(x_coordinate_dict.values()), max(x_coordinate_dict.values())]

    fig_corrected.add_trace(go.Scatter(
        x=dummy_x_data,
        y=[18, 18],
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="skip"
    ))

    # Update layout
    fig_corrected.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    # Update layout to hide original y-axis elements and set ranges
    y_min = min(y_coordinate_dict.values()) - 10
    y_max = max(y_coordinate_dict.values()) + 10
    fig_corrected.update_layout(
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[y_min, 25]
        ),
        xaxis=dict(
            # range=[x_min - 5, x_max + 5]
            range=[min(x_coordinate_dict.values()) - timedelta(days=5),
                   max(x_coordinate_dict.values()) + timedelta(days=5)] if project_start_date else [x_min - 5,
                                                                                                    x_max + 5]
        )
    )

    # Configure x-axis ticks if project_start_date is provided
    if project_start_date:
        start_date = datetime.strptime(project_start_date, '%Y-%m-%d')

        # Generate tick values and labels
        min_date = min(x_coordinate_dict.values())
        max_date = max(x_coordinate_dict.values())

        # Calculate tick positions at daily intervals
        tick_vals =[]
        tick_text =[]
        current_date = min_date
        while current_date <= max_date:
            tick_vals.append(current_date)
            tick_text.append(current_date.strftime('%Y-%m-%d'))  # Format date as string
            current_date += timedelta(days=1)

        # Add a separate trace for the ticks
        fig_corrected.add_trace(go.Scatter(
            x=tick_vals,  # Use tick_vals for x-coordinates
            y=[18] * len(tick_vals),  # Position ticks at the dummy axis level
            mode='markers',  # Use markers for ticks
            marker=dict(symbol='line-ns', size=5, color='black'),  # Style the ticks
            hoverinfo='skip'
        ))

        fig_corrected.update_layout(
            xaxis=dict(
                tickmode='array',  # Use array for custom ticks
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=-45,  # Rotates the ticks by -45 degrees
                showgrid=False,
                zeroline=False,
                # range=[x_min - 5, x_max + 5]
                range=[min(x_coordinate_dict.values()) - timedelta(days=5),
                       max(x_coordinate_dict.values()) + timedelta(days=5)] if project_start_date else [x_min - 5,
                                                                                                        x_max + 5]
            )
        )
    else:
        fig_corrected.update_layout(
            xaxis=dict(
                range=[x_min - 5, x_max + 5]
            )
        )

    # Set the width and height of the plot
    fig_corrected.update_layout(
        width=800,
        height=800
    )

    return fig_corrected

def calculate_time_to_finish(df, activity, x_coordinate_dict, y_coordinate_dict, all_paths_with_durations,
                                 critical_path_duration):
    """
    Calculates the 'time to finish' for an activity.

    Args:
        activity (int): The activity number.
        x_coordinate_dict (dict): Dictionary of x-coordinates for each activity.
        y_coordinate_dict (dict): Dictionary of y-coordinates for each activity.
        all_paths_with_durations (dict): Dictionary containing all paths and their durations.
        critical_path_duration (int): The duration of the critical path.

    Returns:
        int: The calculated time to finish for the activity.
    """
    # Find the critical path
    max_duration = 0
    critical_path = None
    for start_activity, paths in all_paths_with_durations.items():
        for path_data in paths:
            if path_data['duration'] > max_duration:
                max_duration = path_data['duration']
                critical_path = path_data['path']

    id = (critical_path[-1])
    days =  (df[df['Activity'] == id]['Duration'].iloc[0])

    if activity in critical_path:
        # Activity is on the critical path
        #time_to_finish = critical_path_duration - x_coordinate_dict[activity]
        time_to_finish = x_coordinate_dict[activity]
    else:
        # Activity is on a non-critical path (branch)
        # Find the path that includes this activity
        for start_activity, paths in all_paths_with_durations.items():
            for path_data in paths:
                if activity in path_data['path']:
                    path = path_data['path']
                    activity_index = path.index(activity)
                    # Calculate time to finish based on the path and critical path duration
                    time_to_finish = critical_path_duration - days
                    for i in range(len(path) - 1, activity_index, -1):
                        time_to_finish -= df.set_index('Activity').loc[path[i]]['Duration']
                    return time_to_finish  # Return the value as soon as it's calculated

    return time_to_finish
def main():
    st.title("Activity Network Diagram")

    # File Upload
    uploaded_file = st.file_uploader("Upload your Project.csv file", type="csv")

    # Project Start Date Input
    project_start_date = st.text_input("Enter the project's start date (YYYY-MM-DD):")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Predecessors'] = df['Predecessors'].apply(convert_predecessors)

        # Get all sequential paths with durations
        all_paths_with_durations = all_paths_with_duration(df)

        # Find the path with the maximum duration (critical path)
        max_duration = 0
        max_path = None
        for start_activity, paths in all_paths_with_durations.items():
            for path_data in paths:
                if path_data['duration'] > max_duration:
                    max_duration = path_data['duration']
                    max_path = path_data['path']

        # Visualization parameters
        radius = 2
        x_padding = 1  # Adjust this value to add more space
        y_level_spacing = 10.0  # Adjust the spacing between y-levels

        # Collect all activities in paths
        all_activities = set()
        for paths in all_paths_with_durations.values():
            for path_data in paths:
                all_activities.update(path_data['path'])

        # Calculate x-coordinates based on the cumulative sum of predecessors' durations
        x_coordinate_dict = {}
        activity_map = df.set_index('Activity').to_dict('index')

        for activity in all_activities:
            predecessors = activity_map[activity].get('Predecessors')

            if not predecessors:
                x_coordinate_dict[activity] = 0  # Start at 0 if no predecessors
            else:
                cumulative_predecessor_duration = 0
                for predecessor in predecessors:
                    cumulative_predecessor_duration += x_coordinate_dict.get(predecessor, 0) + activity_map[predecessor][
                        'Duration']
                x_coordinate_dict[activity] = cumulative_predecessor_duration

        # Initialize y-coordinates for the critical path
        y_coordinate_dict = {activity: 0 for activity in max_path}

        # Identify branching points
        branching_points = {}
        for activity, data in activity_map.items():
            predecessors = data.get('Predecessors')
            if predecessors:
                for predecessor in predecessors:
                    if predecessor not in branching_points:
                        branching_points[predecessor] = 0
                    branching_points[predecessor] += 1

        # Calculate y-coordinates for other paths
        y_level = 1
        y_direction = 1  # Start with +1 for the first branch

        processed_branches = set()  # Track processed branches to avoid redundant calculations

        for start_activity, paths in all_paths_with_durations.items():
            for path_data in paths:
                path = path_data['path']
                if path != max_path and tuple(path) not in processed_branches:

                    branch_start = path[0]  # Determine the branch start

                    if branching_points.get(branch_start, 0) > 1:  # Only handle branches from branching points

                        # Calculate x-range for the current branch
                        min_x_branch = float('inf')
                        max_x_branch = float('-inf')
                        for activity in path:
                            x_coord = x_coordinate_dict.get(activity)
                            if x_coord is not None:
                                min_x_branch = min(min_x_branch, x_coord)
                                max_x_branch = max(max_x_branch, x_coord)

                        # Assign y-coordinates based on x-ranges and alternating levels
                        for i, activity in enumerate(path):
                            if activity not in y_coordinate_dict:
                                y_coordinate_dict[activity] = y_level * y_direction

                    y_level += 1
                    y_direction *= -1

                processed_branches.add(tuple(path))

        # Create a DataFrame for all activities in the paths
        all_paths_df = df[df['Activity'].isin(all_activities)].reset_index(drop=True)

        # --- Offset Sliders ---
        x_offset = st.slider("X Offset", -1.0, 1.0, 0.25, step=0.05)  # Slider for x-offset
        y_offset = st.slider("Y Offset", -1.0, 1.0, 0.75, step=0.05)  # Slider for y-offset


        # Create the network diagram
        fig = create_network_diagram(all_paths_df, x_coordinate_dict, y_coordinate_dict, radius, x_offset, y_offset,
                                     project_start_date, max_duration, all_paths_with_durations)
        # Update layout limits
        min_x = min(x_coordinate_dict.values()) - radius - x_padding
        # Calculate max_x correctly
        max_x_value = max(x_coordinate_dict.values()) if x_coordinate_dict else 0
        max_activity = max(x_coordinate_dict, key=x_coordinate_dict.get) if x_coordinate_dict else None
        max_duration = activity_map[max_activity]['Duration'] if max_activity else 0
        max_x = max_x_value + max_duration + radius + x_padding
        min_y = min(min(y_coordinate_dict.values()) - radius - 1, -y_level_spacing)
        max_y = max(max(y_coordinate_dict.values()) + radius + 1, y_level_spacing)

        fig.update_layout(
            xaxis=dict(range=[min_x, max_x]),
            yaxis=dict(range=[min_y, max_y])
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # --- Interactive Activity Rearrangement ---
        st.subheader("Rearrange Activities")
        st.write("Modify the Y coordinates in the table below and click 'Update Coordinates'.")

        # Create a DataFrame to display and edit activity coordinates
        edit_df = pd.DataFrame(list(y_coordinate_dict.items()), columns=['Activity', 'Y Coordinate'])
        edit_df = edit_df.set_index('Activity')  # Set 'Activity' as index for easier editing

        # Display the DataFrame for editing
        edited_df = st.data_editor(edit_df)

        if st.button("Update Coordinates"):
            # Update the y_coordinate_dict with the edited values
            for activity, row in edited_df.iterrows():
                y_coordinate_dict[activity] = row['Y Coordinate']

            # --- Redraw the plot with updated coordinates ---
            fig = create_network_diagram(all_paths_df, x_coordinate_dict, y_coordinate_dict, radius, x_offset,
                                        y_offset, project_start_date, max_duration, all_paths_with_durations)

            # Update layout limits
            min_x = min(x_coordinate_dict.values()) - radius - x_padding
            # Calculate max_x correctly
            max_x_value = max(x_coordinate_dict.values()) if x_coordinate_dict else 0
            max_activity = max(x_coordinate_dict,
                               key=x_coordinate_dict.get) if x_coordinate_dict else None
            max_duration = activity_map[max_activity]['Duration'] if max_activity else 0
            max_x = max_x_value + max_duration + radius + x_padding
            min_y = min(min(y_coordinate_dict.values()) - radius - 1, -y_level_spacing)
            max_y = max(max(y_coordinate_dict.values()) + radius + 1, y_level_spacing)

            fig.update_layout(
            xaxis = dict(range=[min_x, max_x]),
            yaxis = dict(range=[min_y, max_y])
        )

        st.plotly_chart(fig, key="network_diagram")  # Add a unique key here
        st.success("Activity coordinates updated.")

if __name__ == "__main__":
    main()
