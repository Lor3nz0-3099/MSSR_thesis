"""Contact-driven TRAIN button verification; writes measured physics evidence."""
import argparse,json,math,pathlib
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=20000);args=parser.parse_args()
from smores_ep.isaac.phase1_manipulation_curriculum import manipulation_curriculum_identity
manipulation_curriculum_identity(args.seed,split='train')
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True);app=launcher.app
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from pxr import Gf,UsdGeom,UsdPhysics,UsdShade
from smores_ep.isaac.phase1_manipulation_curriculum import install_phase1_manipulation_curriculum_course
sim=SimulationContext(sim_utils.SimulationCfg(dt=1/240,device='cpu',use_fabric=False))
stage=sim.stage
material=UsdShade.Material.Define(stage,"/World/materials/dynamic_ground")
friction=UsdPhysics.MaterialAPI.Apply(material.GetPrim());friction.CreateStaticFrictionAttr(.8);friction.CreateDynamicFrictionAttr(.6);friction.CreateRestitutionAttr(0.)
course=install_phase1_manipulation_curriculum_course(stage,seed=args.seed,split='train')
root='/World/Phase1ManipulationCurriculum';path=root+'/Phase1ManipulationButtonPlunger'
normal=course.press_direction_world_xy;center=course.button_center_xyz_m
yaw=math.degrees(math.atan2(normal[1],normal[0]))
pusher=UsdGeom.Cube.Define(stage,'/World/ContactPusher');pusher.CreateSizeAttr(1)
translate=pusher.AddTranslateOp()
pusher.AddRotateZOp().Set(yaw);pusher.AddScaleOp().Set(Gf.Vec3f(.02,.06,.06))
# Face center is 20 mm in front of the plunger centre; initial 1 mm clearance.
start=(center[0]-.031*normal[0],center[1]-.031*normal[1],center[2])
translate.Set(Gf.Vec3d(*start))
UsdPhysics.CollisionAPI.Apply(pusher.GetPrim())
body=UsdPhysics.RigidBodyAPI.Apply(pusher.GetPrim());body.CreateKinematicEnabledAttr(True)
sim.reset()
maximum=0.;samples=[]
try:
 for index in range(240):
  distance=.0048*min(index/180,1)
  translate.Set(Gf.Vec3d(start[0]+distance*normal[0],start[1]+distance*normal[1],start[2]))
  sim.step(render=False)
  position=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
  depression=sum((float(position[k])-center[k])*normal[k] for k in range(2))
  maximum=max(maximum,depression)
  if index%12==0:samples.append({'tick':index,'depression_m':depression})
 result={'schema_version':'mssr.c1_button_physical_smoke.v1','seed':args.seed,'split':'train','physics_hz':240,'actuation':'kinematic_contact_pusher','measured_max_depression_m':maximum,'success':maximum>=.0035,'samples':samples}
 pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
 print('C1_BUTTON_PHYSICAL_RESULT',json.dumps(result),flush=True)
 assert result['success'] and maximum <= .0041, maximum
finally:
 sim.stop();app.close()
