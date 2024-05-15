import pathlib
import random
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from heuristic.alns_wahyu.alns import ALNS_W
from heuristic.alns_wahyu.arguments import prepare_args
from heuristic.alns_wahyu.evaluator.safak_evaluator import SafakEvaluator
from heuristic.alns_wahyu.operator.destroy import RandomRemoval, WorstRemoval
from heuristic.alns_wahyu.operator.repair import GreedyRepair, RandomRepair
from solver.utils import visualize_box   

def setup_destroy_operators(args):
    operator_list = [RandomRemoval(), WorstRemoval()]
    return operator_list
    
def setup_repair_operators(args):
    operator_list = [RandomRepair(), GreedyRepair()]
    return operator_list

def setup_log_writer(args):
    summary_root = "logs"
    summary_dir = pathlib.Path(".")/summary_root
    experiment_summary_dir = summary_dir/args.title
    experiment_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_summary_dir.absolute())
    return writer
    

def plot_convergence_chart(scores, title):
    x = range(len(scores))
    y = scores

    fig, ax = plt.subplots()
    plt.plot(x, y, '-o', color= 'white')

    slope, intercept = np.polyfit(x, y, 1)
    trendline = intercept + slope * np.array(x)

    plt.plot(x, trendline, color='#FF914D')
    # Mengubah warna axes (sumbu) ke putih
    plt.gca().spines['bottom'].set_color('white')

    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')

    # Mengubah warna ticks (penanda pada sumbu) ke putih
    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')
    # Untuk latar belakang hitam, ubah facecolor dari figure dan axes
    plt.gcf().set_facecolor('#245076')
    plt.gca().set_facecolor('#245076')
    fig.savefig("convergence_charts/"+title+".jpg", dpi=fig.dpi)

def plot_destroy_counts(destroy_operators, destroy_count_logs, title):
    plt.close()
    fig, ax = plt.subplots()
    for d_idx in range(len(destroy_operators)):
        destroy_count = [dc[d_idx] for dc in destroy_count_logs]
        x = range(len(destroy_count))
        y = destroy_count
        plt.plot(x,y, label=str(destroy_operators[d_idx]))
    plt.legend()
    fig.savefig("operator_charts/destroy_"+title+".jpg", dpi=fig.dpi)

def plot_repair_counts(repair_operators, repair_count_logs, title):
    plt.close()
    fig, ax = plt.subplots()
    for d_idx in range(len(repair_operators)):
        repair_count = [dc[d_idx] for dc in repair_count_logs]
        x = range(len(repair_count))
        y = repair_count
        plt.plot(x,y, label=str(repair_operators[d_idx]))
    plt.legend()
    fig.savefig("operator_charts/repair_"+title+".jpg", dpi=fig.dpi)

def visualization(visualization_data, title):    
    def get_random_color():
        return "#" + "".join([random.choice('0123456789ABCDEF') for _ in range(6)])

    # Function to draw a cuboid with a dynamic random color
    def draw_cuboid(ax, position, size, alpha=0.3):
        # Generate the corners of a cuboid
        ox, oy, oz = position
        l, w, h = size
        x = [ox, ox, ox+l, ox+l, ox, ox, ox+l, ox+l]
        y = [oy, oy+w, oy+w, oy, oy, oy+w, oy+w, oy]
        z = [oz, oz, oz, oz, oz+h, oz+h, oz+h, oz+h]
        vertices = np.array([[x[i], y[i], z[i]] for i in range(8)])

        # Generate the list of sides' polygons
        verts = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                 [vertices[4], vertices[5], vertices[6], vertices[7]], 
                 [vertices[0], vertices[1], vertices[5], vertices[4]], 
                 [vertices[2], vertices[3], vertices[7], vertices[6]], 
                 [vertices[1], vertices[2], vertices[6], vertices[5]],
                 [vertices[4], vertices[7], vertices[3], vertices[0]]]

        # Generate a random color for this cuboid
        color = get_random_color()

        # Create the 3D polygons and add to the axes
        poly3d = Poly3DCollection(verts, facecolors=color, alpha=alpha)
        ax.add_collection3d(poly3d)

    def draw_scaled_container(ax, dimensions, line_width=0.3, color='black', alpha=0.3):
        L, W, H = dimensions['length'], dimensions['width'], dimensions['height']

        # Points of the container
        points = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],   # bottom
                           [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]])  # top

        # Define the vertices for each face of the container
        faces = [#[points[0], points[1], points[2], points[3]],  # bottom
                 #[points[4], points[5], points[6], points[7]],  # top
                 #[points[0], points[1], points[5], points[4]],  # front
                 [points[2], points[3], points[7], points[6]],  # back
                 [points[1], points[2], points[6], points[5]],  # right
                 [points[0], points[3], points[7], points[4]]]  # left

        # Create a 3D polygon for each face
        container = Poly3DCollection(faces, facecolors=color, linewidths=line_width, edgecolor='r', alpha=alpha)

        # Add the container to the plot
        ax.add_collection3d(container)

    def plot_cargo():
        df = visualization_data

        # Find out how many unique containers there are
        containers = df['Container'].unique()


        for i, container_number in enumerate(containers, start=1):
            plt.close()
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            # Filter the dataframe for the current container
            container_df = df[df['Container'] == container_number]

            if container_df.empty:
                continue

            # Create a subplot for this container
            # ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')

            # container dimensions are the same for all, or you can adjust this if needed
            #container_dimensions = {'length': 12, 'width': 3, 'height': 4}
            container_dimensions = {
                'length': container_df['ContainerLength'].iloc[0],
                'width': container_df['ContainerWidth'].iloc[0],
                'height': container_df['ContainerHeight'].iloc[0]}

            # Draw scaled container
            draw_scaled_container(ax, container_dimensions)

            # Draw cargo items for this container
            for index, row in container_df.iterrows():
                coordinate = (row['X'], row['Y'], row['Z'])
                dimensions = (row['Length'], row['Width'], row['Height'])
                draw_cuboid(ax, coordinate, dimensions)
                # Label at the center of the top face of the cuboid
                ax.text(coordinate[0] + dimensions[0]/2, 
                        coordinate[1] + dimensions[1]/2, 
                        coordinate[2] + dimensions[2], 
                        str(int(row['Item'])), color='red', ha='center', va='bottom')

            # Set the view angle
            ax.view_init(elev=38, azim=215)

            # Set labels and axes limits with the same scale
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            max_dimension = max(container_dimensions.values())
            ax.set_xlim(0, max_dimension)
            ax.set_ylim(0, max_dimension)
            ax.set_zlim( max_dimension,0)
            
            ax.invert_zaxis()

            # Set the title of each subplot to the container number
            ax.set_title(f'Container {round(container_number)}')

            # Display the figure with all container plots
            plt.tight_layout()
            fig.savefig("container_figs/"+title+"_container -"+str(i)+".jpg", dpi=fig.dpi)
    plot_cargo()


def run(args):
    data_path = pathlib.Path()/"instances"/"data_from_wahyu"/args.instance_filename
    df_cargos = pd.read_excel(data_path.absolute(),sheet_name='Item', header=0)
    df_containers = pd.read_excel(data_path.absolute(),sheet_name='Container', header=0)
    destroy_operators = setup_destroy_operators(args)
    repair_operators = setup_repair_operators(args)
    log_writer = setup_log_writer(args)
    alns_solver = ALNS_W(destroy_operators,
                         repair_operators,
                         SafakEvaluator(args.insertion_mode, args.cargo_sort_criterion),
                         log_writer,
                         args.max_iteration,
                        #  args.max_feasibility_repair_iteration,
                         args.omega,
                         args.a,
                         args.b1,
                         args.b2,
                         args.d1,
                         args.d2)
    start = time.time()
    alns_solver.solve(df_cargos, df_containers)
    end = time.time()
    total_computation_time = end - start
    print('total computation time', total_computation_time)


    current_result, best_result = alns_solver.current_eval_result, alns_solver.best_eval_result
    best_iteration = alns_solver.best_iteration
    print('Overall utility',best_result.overall_utility)
    print('Utility per container',best_result.container_utilities)
    print('Best iteration', best_iteration)
    print('Best solution',np.array2string(best_result.x,threshold=np.inf))
    # print('Best solution:')
    # for ccm in best_result.x:
    #     for cargo_state in ccm:
    #         print(cargo_state, end=" ")
    #     print()
    # plot_destroy_counts(alns_solver.destroy_operators, alns_solver.destroy_count_logs, args.title)
    # plot_repair_counts(alns_solver.repair_operators, alns_solver.repair_count_logs, args.title)
    print('Total item', best_result.num_cargo_packed)
    print('Best fitness',best_result.score)
    print("cargo vol", best_result.x.dot(best_result.cargo_volumes), best_result.x.dot(best_result.cargo_weights))
    for d_idx in range(len(alns_solver.destroy_operators)):
        print(str(destroy_operators[d_idx])+" Count", alns_solver.destroy_counts[d_idx])
    for r_idx in range(len(alns_solver.repair_operators)):
        print(str(repair_operators[r_idx])+" Count", alns_solver.repair_counts[r_idx])
    # plot_convergence_chart(alns_solver.best_scores, args.title)    

    for s_idx, solution in enumerate(best_result.solution_list):
        print("--------Container -",s_idx,":")
        print(solution)


    
    # visualization_info_list = []
    # for s_idx, solution in enumerate(best_result.solution_list):
    #     if len(solution.positions)==0:
    #         continue
    #     real_dims = solution.real_cargo_dims
    #     container_dim = solution.container_dims[0,:]
    #     for c_idx in range(len(solution.positions)):
    #         info = {}
    #         position = solution.positions[c_idx]
    #         real_dim = real_dims[c_idx]
    #         info["Item"] = solution.cargo_type_ids[c_idx]
    #         info["Container"] = s_idx
    #         info["X"] = position[0]
    #         info["Y"] = position[1]
    #         info["Z"] = position[2]
    #         info["Length"] = real_dim[0]
    #         info["Width"] = real_dim[1]
    #         info["Height"] = real_dim[2]
    #         info["ContainerLength"] = container_dim[0]
    #         info["ContainerWidth"]  = container_dim[1]
    #         info["ContainerHeight"] = container_dim[2]
    #         visualization_info_list += [info]
    # visualization_df = pd.DataFrame(visualization_info_list, columns=["Item",	"Container",	"X",	"Y",	"Z",	"Length",	"Width",	"Height",	"ContainerLength",	"ContainerWidth",	"ContainerHeight"])
    # visualization(visualization_df, args.title)

    return best_result, current_result, best_iteration, alns_solver,total_computation_time


import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import matplotlib.pyplot as plt
import tempfile
import os
import io
import sys
import re

def save_plot(plot_func):
    """Function to save plot to a temporary file and return the path"""
    # Use delete=False to ensure the file is not deleted automatically
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        plt.figure()
        plot_func()
        plt.savefig(tmpfile.name)
        plt.close()
        return tmpfile.name

def parse_details(details):
    # Membuat dictionary untuk menyimpan hasil ekstraksi
    result = {}
    
    # Memisahkan setiap baris dan mengiterasi melalui setiap baris untuk mengekstrak data
    lines = details.split('\n')
    for line in lines:
        if "Total volume packed:" in line:
            result['Total_volume_packed'] = float(line.split(':')[1].split('(')[0].strip())
        elif "Profit:" in line:
            result['Profit'] = float(line.split(':')[1].strip())
        elif "Revenue:" in line:
            result['Revenue'] = float(line.split(':')[1].strip())
        elif "Expense:" in line:
            result['Expense'] = float(line.split(':')[1].strip())
        elif "Center of gravity:" in line:
            # Menghapus kurung siku dan memisahkan nilai berdasarkan spasi
            result['Central_of_gravity'] = [float(x) for x in line.split(':')[1].replace('[', '').replace(']', '').strip().split()]

    return result


if __name__ == "__main__":
    args = prepare_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    # matplotlib.use("Agg")
    run(args)
    exit()
    
    excel_filename = args.title+'_result.xlsx'  # Explicit filename for ease of use
    temp_files = []  # List to track temporary files
    rep=args.num_replication

    recap_data = {
        'Replication': [],
        'Best Fitness': [],
        'Total Computation Time': [],
        'Overall Utilization': [],
        'Total Volume Packed': [],
        'Profit': [],
        'Revenue': [],
        'Expenses': [],
        'Center of Gravity': []  # Assuming this is a single representative value per replication
    }
    # Write data using pandas
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        for i in range(rep):  # Looping 10 times
            print(f"Running replication {i+1}")
            recap_data['Replication'].append(f'Replication {i+1}')
            best_result, current_result, best_iteration, alns_solver,total_computation_time = run(args)
            destroy_counts = []
            repair_counts = []
            # Collect data from destroy operators
            for d_idx in range(len(alns_solver.destroy_operators)):
                destroy_counts.append(alns_solver.destroy_counts[d_idx])

            # Collect data from repair operators
            for r_idx in range(len(alns_solver.repair_operators)):
                repair_counts.append(alns_solver.repair_counts[r_idx])

            container_solutions = []

            for s_idx, solution in enumerate(best_result.solution_list):
                # Redirect stdout to capture print statements
                old_stdout = sys.stdout  # Save the old stdout
                result = io.StringIO()
                sys.stdout = result
                
                # This will trigger the __str__ method of the solution, which prints the details
                print(solution)  
                
                # Restore old stdout
                sys.stdout = old_stdout
                
                # Store the captured output and other details
                solution_details = result.getvalue()
                result.close()

                container_solutions.append({
                    "container_index": s_idx,
                    "details": solution_details
                })

            Rev = []
            Exp = []
            Prof = []
            Vol = []
            CoG = []

            # Iterasi melalui setiap container dalam list 'a'
            for container in container_solutions:
                info = parse_details(container['details'])
                # Menambahkan data ke masing-masing list
                Rev.append(info.get('Revenue', 0))  # Default ke 0 jika tidak ditemukan
                Exp.append(info.get('Expense', 0))
                Prof.append(info.get('Profit', 0))
                Vol.append(info.get('Total_volume_packed', 0))
                CoG.append(info.get('Central_of_gravity', [0, 0]))  # Default ke [0, 0] jika tidak ditemukan

            Rev = np.sum(Rev)
            Exp = np.sum(Exp)
            Prof = np.sum(Prof)
            # Collect data into a DataFrame
            data = {
                'Overall Utility': [best_result.overall_utility],
                'Utility Per Container': [best_result.container_utilities],
                'Total Item Packed': [best_result.num_cargo_packed],
                'Best Fitness': [best_result.score],
                'Best iteration': [best_iteration],
                'Computational time': [total_computation_time],
                'Best Solution': [np.array2string(best_result.x,threshold=np.inf)],
                'Random removal': [destroy_counts[0]],
                'Worst removal' : [destroy_counts[1]],
                'Random reapair' : [repair_counts[0]],
                'Greedy insertion' : [repair_counts[1]],
                'Container info' : [container_solutions],
                'Revenue': [Rev],
                'Expenses': [Exp],
                'Profit': [Prof],
                'Vol packed': [Vol],
                'COG': [CoG]
            }
            

            # Append data for this replication to the recap_data
            recap_data['Best Fitness'].append(best_result.score)
            recap_data['Total Computation Time'].append(total_computation_time)
            recap_data['Overall Utilization'].append(best_result.overall_utility)
            recap_data['Total Volume Packed'].append((Vol))
            recap_data['Profit'].append((Prof))
            recap_data['Revenue'].append((Rev))
            recap_data['Expenses'].append((Exp))
            # Calculate a single representative value for Center of Gravity, if necessary
            recap_data['Center of Gravity'].append((CoG))
            
            visualization_info_list = []
            for s_idx, solution in enumerate(best_result.solution_list):
                if len(solution.positions)==0:
                    continue
                real_dims = solution.real_cargo_dims
                container_dim = solution.container_dims[0,:]
                for c_idx in range(len(solution.positions)):
                    info = {}
                    position = solution.positions[c_idx]
                    real_dim = real_dims[c_idx]
                    info["Item"] = solution.cargo_type_ids[c_idx]
                    info["Container"] = s_idx
                    info["X"] = position[0]
                    info["Y"] = position[1]
                    info["Z"] = position[2]
                    info["Length"] = real_dim[0]
                    info["Width"] = real_dim[1]
                    info["Height"] = real_dim[2]
                    info["ContainerLength"] = container_dim[0]
                    info["ContainerWidth"]  = container_dim[1]
                    info["ContainerHeight"] = container_dim[2]
                    visualization_info_list += [info]
            visualization_df = pd.DataFrame(visualization_info_list, columns=["Item",	"Container",	"X",	"Y",	"Z",	"Length",	"Width",	"Height",	"ContainerLength",	"ContainerWidth",	"ContainerHeight"])

            # Data for each replication to 
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f'Replication {i+1}')
            visualization_df.to_excel(writer, sheet_name=f'Replication {i+1}', startrow=len(df)+5)

    with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a') as writer:
        recap_df = pd.DataFrame(recap_data)
        recap_df.to_excel(writer, sheet_name='Summary', index=False)

    # Step 2: Open with openpyxl to add images
    visualization_info_list = []
    for s_idx, solution in enumerate(best_result.solution_list):
        if len(solution.positions)==0:
            continue
        real_dims = solution.real_cargo_dims
        container_dim = solution.container_dims[0,:]
        for c_idx in range(len(solution.positions)):
            info = {}
            position = solution.positions[c_idx]
            real_dim = real_dims[c_idx]
            info["Item"] = solution.cargo_type_ids[c_idx]
            info["Container"] = s_idx
            info["X"] = position[0]
            info["Y"] = position[1]
            info["Z"] = position[2]
            info["Length"] = real_dim[0]
            info["Width"] = real_dim[1]
            info["Height"] = real_dim[2]
            info["ContainerLength"] = container_dim[0]
            info["ContainerWidth"]  = container_dim[1]
            info["ContainerHeight"] = container_dim[2]
            visualization_info_list += [info]
    visualization_df = pd.DataFrame(visualization_info_list, columns=["Item",	"Container",	"X",	"Y",	"Z",	"Length",	"Width",	"Height",	"ContainerLength",	"ContainerWidth",	"ContainerHeight"])
    
    wb = load_workbook(excel_filename)
    unique_containers = visualization_df['Container'].unique()
    for i in range(rep):
        destroy_counts_path = save_plot(lambda: plot_destroy_counts(alns_solver.destroy_operators, alns_solver.destroy_count_logs, f"{args.title} Rep{i+1}"))
        repair_counts_path = save_plot(lambda: plot_repair_counts(alns_solver.repair_operators, alns_solver.repair_count_logs, f"{args.title} Rep{i+1}"))
        convergence_chart_path = save_plot(lambda: plot_convergence_chart(alns_solver.best_scores, f"{args.title} Rep{i+1}"))
        # visualization_pict_path =save_plot(lambda: visualization(visualization_df, f"{args.title} Rep{i+1}"))

        ws = wb[f'Replication {i+1}']
        img1 = Image(destroy_counts_path)
        img2 = Image(repair_counts_path)
        img3 = Image(convergence_chart_path)
        # img4 = Image(visualization_pict_path)
        ws.add_image(img1, 'A1')
        ws.add_image(img2, 'E5')
        ws.add_image(img3, 'I10')
        # ws.add_image(img4, 'K15')  # Adjust position as needed
        for container_idx in unique_containers:
        # Filter data untuk container saat ini
            current_container_df = visualization_df[visualization_df['Container'] == container_idx]
            
            # Jika tidak ada data untuk container saat ini, lanjutkan ke container berikutnya
            if current_container_df.empty:
                continue

            # Buat visualisasi untuk container saat ini
            visualization_pict_path = save_plot(lambda: visualization(current_container_df,f"{args.title} Rep{i+1}"))

            # Tambahkan gambar visualisasi ke workbook
            img4 = Image(visualization_pict_path)
            ws.add_image(img4, 'K30')

    wb.save(excel_filename)
    # Clean up temporary files after saving the workbook
    for file_path in temp_files:
        os.remove(file_path)

    print("All replications are completed and saved.")
    
    