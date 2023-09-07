	def weld_segment_tri(self,primitives,robot1,positioner,robot2,q1_all,positioner_all,q2_all,v1_all,v2_all,cond_all,arc=False,cool=False,wait=0):
		###robot+positioner weld segment, MOVEJ + MOVEL x (N-1)
		#q1_all: list of robot joint angles (N x 6)
		#positioner_all: list of positioner joint angles (N x 2)
		#q2_all: list of positioenr joint angles (N x 6)
		#v1_all: list of robot1segment speed (N x 1)
		#v2_all: list of robot2segment speed (N x 1)
		#cond_all: list of job number (N x 1) or (1,), 0 refers to off

		q1_all=np.degrees(q1_all)
		positioner_all=np.degrees(positioner_all)
		q2_all=np.degrees(q2_all)
		arcof=True
		coolof=True
		
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[positioner.robot_name],
		   ROBOT_CHOICE3=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=positioner.pulse2deg,pulse2deg_3=robot2.pulse2deg, 
		   tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		
		mp.primitive_call_tri(primitives[0],q1_all[0],v1_all[0],target2=['MOVJ',positioner_all[0],None],target3=['MOVJ',q2_all[0],v2_all[0]])
		if arc:
			if len(cond_all)==1: 
				mp.setArc(True,cond_num=cond_all[0])
				arcof=False
			else:
				if cond_all[1]!=0:
					mp.setArc(True,cond_num=cond_all[1])
					arcof=False


		for i in range(1,len(q1_all)):
			if len(cond_all)>1 and arc and i>1:
				if arcof:
					if cond_all[i]!=0:
						mp.setArc(True, cond_all[i])
						arcof=False
				else:
					if cond_all[i]==0:
						mp.setArc(False)
						arcof=True
					elif cond_all[i]!=cond_all[i-1]:
						mp.changeArc(cond_all[i])
						
		if cool:
			if len(cond_all)==1: 
				mp.setDO(21,True)
				coolof=False
			else:
				if cond_all[1]!=0:
					mp.setDO(21,True)
					coolof=False


		for i in range(1,len(q2_all)):
			if len(cond_all)>1 and cool and i>1:
				if coolof:
					if cond_all[i]!=0:
						mp.setDO(21,True)
						coolof=False
				else:
					if cond_all[i]==0:
						mp.setDO(21,False)
						coolof=True						

			mp.primitive_call_tri(primitives[i],q1_all[i],v1_all[i],target2=['MOVJ',positioner_all 	[i],None],target3=['MOVL',q2_all[i],v2_all[i]])
		
        
        if cool and not coolof:
			mp.setDO(21,False)			
		if arc and not arcof:
			mp.setArc(False)
		return self.client.execute_motion_program(mp)