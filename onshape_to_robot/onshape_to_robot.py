import numpy as np
from copy import copy
import commentjson as json
from colorama import Fore, Back, Style
import sys
from sys import exit
import os
import hashlib
from . import csg
from .robot_description import RobotURDF, RobotSDF
import re

partNames = {}

def main():
    # Loading configuration, collecting occurrences and building robot tree
    from .load_robot import \
        config, client, tree, occurrences, getOccurrence, frames


    # Creating robot for output
    if config['outputFormat'] == 'urdf':
        robot = RobotURDF(config['robotName'])
    elif config['outputFormat'] == 'sdf':
        robot = RobotSDF(config['robotName'])
    else:
        print(Fore.RED + 'ERROR: Unknown output format: ' +
            config['outputFormat']+' (supported are urdf and sdf)' + Style.RESET_ALL)
        exit()
    robot.drawCollisions = config['drawCollisions']
    robot.jointMaxEffort = config['jointMaxEffort']
    robot.mergeMeshes = config['mergeMeshes']
    robot.maxMeshSize = config['maxMeshSize']
    robot.simplifyMeshes = config['simplifyMeshes']
    robot.jointMaxVelocity = config['jointMaxVelocity']
    robot.noDynamics = config['noDynamics']
    robot.packageName = config['packageName']
    robot.addDummyBaseLink = config['addDummyBaseLink']
    robot.robotName = config['robotName']
    robot.additionalXML = config['additionalXML']
    robot.useFixedLinks = config['useFixedLinks']
    robot.meshDir = config['outputDirectory']


    def partIsIgnore(name):
        if config['whitelist'] is None:
            return name in config['ignore']
        else:
            return name not in config['whitelist']

    # Adds a part to the current robot link


    def addPart(occurrence, matrix):
        part = occurrence['instance']

        if part['suppressed']:
            return

        if part['partId'] == '':
            print(Fore.YELLOW + 'WARNING: Part '+part['name']+' has no partId'+Style.RESET_ALL)
            return

        # Importing mesh file for this part
        justPart, prefix = extractPartName(part['name'], part['configuration'])
        prefix = re.sub(r'[^\/0-9a-zA-Z_-]', '_', prefix)

        extra = ''
        if occurrence['instance']['configuration'] != 'default':
            extra = Style.DIM + ' (configuration: ' + \
                occurrence['instance']['configuration']+')'
        symbol = '+'
        if partIsIgnore(justPart):
            symbol = '-'
            extra += Style.DIM + ' / ignoring visual and collision'

        print(Fore.GREEN + symbol+' Adding part ' +
            occurrence['instance']['name']+extra + Style.RESET_ALL)

        if partIsIgnore(justPart):
            glbFile = None
        else:
            glbFile = prefix.replace('/', '_')+'.glb'
            # shorten the configuration to a maximum number of chars to prevent errors. Necessary for standard parts like screws
            if len(part['configuration']) > 40:
                shortend_configuration = hashlib.md5(
                    part['configuration'].encode('utf-8')).hexdigest()
            else:
                shortend_configuration = part['configuration']
            glb = client.part_glb(part['documentId'], part['documentMicroversion'], part['elementId'],
                                        part['partId'], shortend_configuration)
        if glb == None:
            glbFile = None
        else:
            with open(config['outputDirectory']+'/'+glbFile, 'wb') as stream:
                stream.write(glb)

            partMetadata = prefix.replace('/', '_')+'.part'
            with open(config['outputDirectory']+'/'+partMetadata, 'w', encoding="utf-8") as stream:
                json.dump(part, stream, indent=4, sort_keys=True)

            glbFile = config['outputDirectory']+'/'+glbFile

        # Import the SCAD files pure shapes
        shapes = None
        if config['useScads']:
            scadFile = prefix+'.scad'
            if os.path.exists(config['outputDirectory']+'/'+scadFile):
                shapes = csg.process(
                    config['outputDirectory']+'/'+scadFile, config['pureShapeDilatation'])

        # Obtain metadatas about part to retrieve color
        if config['color'] is not None:
            color = config['color']
        else:
            metadata = client.part_get_metadata(
                part['documentId'], part['documentMicroversion'], part['elementId'], part['partId'], part['configuration'])
            if metadata != None and 'appearance' in metadata:
                colors = metadata['appearance']['color']
                color = np.array(
                    [colors['red'], colors['green'], colors['blue']])/255.0
            else:
                color = [0.5, 0.5, 0.5]

            # XXX: There must be a better way to retrieve the part color
            for entry in metadata['properties']:
                if 'value' in entry and type(entry['value']) is dict and 'color' in entry['value']:
                    rgb = entry['value']['color']
                    color = np.array(
                        [rgb['red'], rgb['green'], rgb['blue']])/255.0

        # Obtain mass properties about that part
        if config['noDynamics']:
            mass = 0
            com = [0]*3
            inertia = [0]*12
        else:
            if prefix in config['dynamicsOverride']:
                entry = config['dynamicsOverride'][prefix]
                mass = entry['mass']
                com = entry['com']
                inertia = entry['inertia']
            else:
                massProperties = client.part_mass_properties(
                    part['documentId'], part['documentMicroversion'], part['elementId'], part['partId'], part['configuration'])

                if massProperties == None or part['partId'] not in massProperties['bodies']:
                    print(Fore.YELLOW + 'WARNING: part ' +
                        part['name']+' has no dynamics (maybe it is a surface)' + Style.RESET_ALL)
                    return
                massProperties = massProperties['bodies'][part['partId']]
                if 'mass' not in massProperties:
                    # The part of type Surface has no mass properties.
                    mass = 0
                    com = [0] * 3
                    inertia = [0] * 9
                else:
                    mass = massProperties["mass"][0]
                    com = massProperties["centroid"]
                    inertia = massProperties["inertia"]

                if abs(mass) < 1e-9:
                    print(Fore.YELLOW + 'WARNING: part ' +
                        part['name']+' has no mass, maybe you should assign a material to it ?' + Style.RESET_ALL)

        pose = occurrence['transform']
        if robot.relative:
            pose = np.linalg.inv(matrix)*pose

        robot.addPart(pose, glbFile, mass, com, inertia, color, shapes, prefix)


    partNames = {}


    def extractPartName(name, configuration):
        parts = name.split(' ')
        del parts[-1]
        basePartName = '_'.join(parts).lower()

        # only add configuration to name if its not default and not a very long configuration (which happens for library parts like screws)
        if configuration != 'default' and len(configuration) < 40:
            parts += ['_' + configuration.replace('=', '_').replace(' ', '_')]

        return basePartName, '_'.join(parts).lower()


    def processPartName(name, configuration, overrideName=None):
        if overrideName is None:
            global partNames
            _, name = extractPartName(name, configuration)

            if name in partNames:
                partNames[name] += 1
            else:
                partNames[name] = 1

            if partNames[name] == 1:
                return name
            else:
                return name+'_'+str(partNames[name])
        else:
            return overrideName


    def buildRobot(tree, matrix):
        occurrence = getOccurrence([tree['id']])
        instance = occurrence['instance']
        print(Fore.BLUE + Style.BRIGHT +
            '* Adding top-level instance ['+instance['name']+']' + Style.RESET_ALL)

        # Build a part name that is unique but still informative
        link = processPartName(
            instance['name'], instance['configuration'], occurrence['linkName'])

        # Create the link, collecting all children in the tree assigned to this top-level part
        robot.startLink(link, matrix)
        for occurrence in occurrences.values():
            if occurrence['assignation'] == tree['id'] and occurrence['instance']['type'] == 'Part':
                addPart(occurrence, matrix)
        robot.endLink()

        # Adding the frames (linkage is relative to parent)
        if tree['id'] in frames:
            for name, part in frames[tree['id']]:
                frame = getOccurrence(part)['transform']
                if robot.relative:
                    frame = np.linalg.inv(matrix)*frame
                robot.addFrame(name, frame)

        # Following the children in the tree, calling this function recursively
        k = 0
        for child in tree['children']:
            worldAxisFrame = child['axis_frame']
            zAxis = child['z_axis']
            jointType = child['jointType']
            jointLimits = child['jointLimits']

            if robot.relative:
                axisFrame = np.linalg.inv(matrix)*worldAxisFrame
                childMatrix = worldAxisFrame
            else:
                # In SDF format, everything is expressed in the world frame, in this case
                # childMatrix will be always identity
                axisFrame = worldAxisFrame
                childMatrix = matrix

            subLink = buildRobot(child, childMatrix)
            robot.addJoint(jointType, link, subLink, axisFrame,
                        child['dof_name'], jointLimits, zAxis)

        return link


    # Start building the robot
    buildRobot(tree, np.matrix(np.identity(4)))
    robot.finalize()
    # print(tree)

    print("\n" + Style.BRIGHT + "* Writing " +
        robot.ext.upper()+" file" + Style.RESET_ALL)
    with open(config['outputDirectory']+'/robot.'+robot.ext, 'w', encoding="utf-8") as stream:
        stream.write(robot.xml)

    if len(config['postImportCommands']):
        print("\n" + Style.BRIGHT + "* Executing post-import commands" + Style.RESET_ALL)
        for command in config['postImportCommands']:
            print("* "+command)
            os.system(command)


if __name__ == "__main__":
    main()