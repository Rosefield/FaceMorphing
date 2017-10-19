import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import image as image_helper
import utils

def plot_mesh(mesh, show_points = False, filename = None):
    """
        Plots the given mesh

        @param mesh, the mesh to plot

        @param show_points, optionally add labels to show the point order

        @param filename, save the mesh at this file
    """
    ps = mesh.points
    
    xmax, ymax = np.max(ps, axis=0)

    fig = plt.figure()
    fig_plt = fig.add_subplot(111)
    fig_plt.triplot(ps[:, 0], ymax - ps[:, 1], mesh.simplices.copy())

    fig.show()

    if show_points:
        for i,p in enumerate(ps):
            fig_plt.text(p[0] -.03, ymax - p[1] + .03, i)

    if utils.save_files and filename is not None:
        plt.savefig(filename)

    input("press enter to continue")

    plt.close(fig)

    return



def delaunay_triangulation(points):
    """
        Returns the Delaunay triangulation of the given points
    """
    return Delaunay(points)

def load_mesh(filename):
    try:

        points = np.loadtxt(filename)

        return delaunay_triangulation(points)
    except:
        return None

def save_mesh(mesh, filename):
    """
        Saves the mesh information in the given filename
    """
    points = mesh.points
    np.savetxt(filename, points)

def get_mesh_matching_points(mesh1, mesh2):
    """
        Returns an (N, 2) array of indices for corresponding points between mesh1 and mesh2.

        Right now this assumes that the points in each are already pre-ordered. Future work could be to use an algorithm like Iterative Closest Point (ICP) to automatically match points. The problem with just choosing the nearest point is that that can result in a many:1 relation of points and we need it to be 1:1 for the algorithm to work properly.

    """

    p1s = mesh1.points
    p2s = mesh2.points

    matches = []

    for i, p in enumerate(p1s):
        matches.append([i,i])
        """
        dp2s = np.linalg.norm(p2s - p, axis=1)
        print(dp2s)
        match = np.argmin(dp2s)
        matches.append([i, match])
        """
    
    matches = np.array(matches)

    return matches

def animate_image_interpolation(im1, im2, mesh1, mesh2, internal_points = 3):
    """
        Creates a set of image interpolations "animating" the morphing of one to the other

        @param im1, im2, the source and destination images, respectively. Each image must have the same dimension

        @param mesh1, mesh, the source and destination meshes, respectively. Each mesh must have the same number of points. Additionally, the points in each mesh should be ordered to match.

        @param internal_points, the number of internal points to use in the interpolation, a total of n+2 images will be returned (including endpoints)

    """

    if(im1.shape != im2.shape):
        print('Images must be of the same size')
        return

    if(mesh1.points.shape != mesh2.points.shape):
        print('Meshes must have the same number of points')
        return

    alphas = np.linspace(1, 0, internal_points + 2, endpoint=True)

    matches = get_mesh_matching_points(mesh1, mesh2)

    src_points = mesh1.points[matches[:,0]]
    dst_points = mesh2.points[matches[:,1]]

    for alpha in alphas:
        mesh = interpolate_meshes(src_points, dst_points, alpha)
        src_face = warp_face(im1, src_points, mesh)
        dst_face = warp_face(im2, dst_points, mesh)
        #print(alpha)
        #image_helper.pick_points(src_face)
        #image_helper.pick_points(dst_face)
        
        image = average_face(src_face, dst_face, alpha)
        
        yield {'mesh':mesh, 'image':image, 'alpha': alpha}


def interpolate_meshes(src_points, dst_points, alpha):
    """
        Create the alpha-average of two meshes. Src and dst points should be pre-sorted so that features match each other
    """
    avg_point = alpha*src_points + (1 - alpha)*dst_points

    mesh = delaunay_triangulation(avg_point)
    
    return mesh

def affine_transforms(simplices, src_points, dst_points):
    """
        For each simplex, create a matrix that maps a point in the src mesh to a point in the dst mesh.
    """
    ones = np.ones(3)
    for simplex in simplices:
        src_tri = np.vstack((src_points[simplex, :].T , ones))
        dst_tri = np.vstack((dst_points[simplex, :].T , ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat
    
def interpolate_image(im, coords):
    """
        Calculate the bilinear interpolation of image pixels around coordinates
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords
    
    y,x = im.shape[:2]
    
    # Make sure that x/y+1 doesn't exceed image dimension
    x0[x0 >= x -1] = x -2
    y0[y0 >= y -1] = y -2

    # 4 Neighour pixels
    q11 = im[y0, x0]
    q21 = im[y0, x0+1]
    q12 = im[y0+1, x0]
    q22 = im[y0+1, x0+1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T 

def warp_face(im, src_points, dst_mesh):
    """
        Warps the given image specified by the src points to math with the destination mesh
    """
    
    image = np.zeros(im.shape, dtype=np.uint8)
    

    affines = np.array(list(affine_transforms(dst_mesh.simplices, src_points, dst_mesh.points)))

    xs, ys = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    positions = np.column_stack([xs.ravel(), ys.ravel()])

    mesh_simplex_indices = dst_mesh.find_simplex(positions)

    for simplex_idx in range(dst_mesh.simplices.shape[0]):
        coords = positions[mesh_simplex_indices == simplex_idx]
        num = len(coords)
        out_coords = np.dot(affines[simplex_idx], np.vstack((coords.T, np.ones(num))))

        x,y = coords.T

        image[y, x] = interpolate_image(im, out_coords)

    
    return image

def average_face(src, dst, alpha):
    im =  (alpha * src) + ((1-alpha)*dst)
    return im.astype(np.uint8)
