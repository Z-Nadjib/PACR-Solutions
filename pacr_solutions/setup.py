from setuptools import find_packages, setup

package_name = 'pacr_solutions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/test_path_planning.launch.xml',
            'launch/test_path_following.launch.xml',
            'launch/test_task_planning.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ai',
    maintainer_email='ai@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_planning = pacr_solutions.path_planning:main',
            'path_following = pacr_solutions.path_following:main',
            'dwa_path_following = pacr_solutions.dwa_path_following:main',
            'task_planning = pacr_solutions.task_planning:main',
        ],
    },
)
