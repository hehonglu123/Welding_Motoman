from setuptools import setup

setup(
    name='motion_primitive',
    version='0.0.1',
    description='Motion Primitive Project Tools',
    url='https://github.com/hehonglu123/Motion-Primitive-Planning',
    py_modules=['blending','error_check','lambda_calc','MotionSend','robots_def','toolbox_circular_fit','utils','realrobot','dual_arm','tes_env','gazebo_model_resource_locator'],
    install_requires=[
        'bs4',
        'numpy',
        'pandas',
        'general_robotics_toolbox',
        'matplotlib',
        'sklearn'
    ]
)