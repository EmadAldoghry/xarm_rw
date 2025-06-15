from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rw'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*'))),
        # Mesh files
        (os.path.join('share', package_name, 'meshes'),
         glob(os.path.join('meshes', '*'))),
        # Model files
        (os.path.join('share', package_name, 'model'),
         glob(os.path.join('model', '*'))),
        # RViz config files
        (os.path.join('share', package_name, 'rviz'),
         glob(os.path.join('rviz', '*'))),

        (os.path.join('share', package_name, 'srdf'),
         glob(os.path.join('srdf', '*'))),
        # World files
        (os.path.join('share', package_name, 'worlds'),
         glob(os.path.join('worlds', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aldoghry',
    maintainer_email='aldoghry@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'data_fusion = rw.fusion_node:main',
            #'viewer = rw.lidar_camera_viewer:main'
            #'depth_camera_viewer = rw.depth_viewer:main',
            #'yellow_object_detector = rw.yellow_follower:main',
            'yellow_object_detector = rw.yellow_object_detector:main',
            'yellow_object_follower = rw.yellow_object_follower:main',
            'chase_the_crack = rw.chase_the_crack:main',
            'joy_c = rw.joy_controller:main',
            'moveit_cartesian_planner = rw.moveit_cartesian_planner:main',
            'crack_path_node = rw.crack_path_node:main',
            'fake_poses = rw.fake_poses:main',
            
        ],
    },
)