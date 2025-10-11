/*********************************************************
                  Main Window!

Click the "Run" button to Run the simulation.

Change the geometry, flow conditions, numerical parameters
visualizations and measurements from this window.

This screen has an example. Other examples are found at 
the top of each tab. Copy/paste them here to run, but you 
can only have one setup & run at a time.

*********************************************************/

import java.util.Random;
BodyUnion bodyunion;
BDIM flow;
FloodPlot flood;
SaveVectorFieldFromBoundary data;
EllipseBody body, body2;
String angle_list;
float t = 0., u = 0.;
 int iter = 0, max_iter = 200;
int stime = 200, etime = 600;
int period = 200;
int sim_length = etime;
float[] angles, angles_full;
int sampling_step=10;
int n=(int)pow(2,7);                       // number of grid points
float xlowerBound = 0.2 * n;
float xupperBound = 0.2 * n;
float ylowerBound = 0.5 * n;
float yupperBound = 0.5 * n;
float dclowerBound = 0.2;
float dcupperBound = 0.8;
float theta0lowerBound = PI / 9; // 20
float theta0upperBound = 2 * PI/ 9; // 40
float theta0plusthetaAupperBound = PI / 3; // 60
float theta0minusthetaAlowerBound = PI / 18; // 10
float thetaAlowerBound = PI / 36;
String root_dir = "Your_path_to_save_data/"; // please change this to your own path
String output_states_path = root_dir + "states/";
String output_force_path = root_dir  + "forces/";
String output_bdry_path = root_dir + "bdry/";

void setup(){
  size(700,700);                             // display window size
}

float[] rotate_angle(int length, int period, float theta_a, float dc) {
  angles = new float[period];  
  angles_full = new float[length];
  for (int i = 0; i < (int)(period * dc); i++) {
    angles[i] = -theta_a * PI / (dc * period) * sin(PI * i / (dc * period));
  }
  for (int i = (int)(period * dc); i < period; i++) {
     angles[i] = theta_a * PI / ((1 - dc)* period) * sin(PI * (i - dc * period)/ ((1 - dc)* period));
  }
  for (int i = 0; i < length; i++) {
    angles_full[i] = angles[i % period];
  }
  return angles_full;
}

float[] rotate_angle_discrete(int length, int period, String angle_list) {
  angles = new float[period];  
  angles_full = new float[length];
  angle_list = angle_list.replaceAll("\\[|\\]|\\s", "");
  String[] elements = split(angle_list, ',');
  println(elements.length);
  for (int i = 0; i < elements.length; i++) {
    angles[i] = float(elements[i]);
  }
  println(angles);

  for (int i = 0; i < length; i++) {
    angles_full[i] = angles[i % period];
  }
  return angles_full;
}

void customsetup(int iter){  
  // Create a Random object
  size(700,700);                             // display window size
  Window view = new Window(n,n);
  Random random = new Random();
  float xrandomFloat = xlowerBound + random.nextFloat() * (xupperBound - xlowerBound);
  float yrandomFloat = ylowerBound + random.nextFloat() * (yupperBound - ylowerBound);
  float dc = dclowerBound + random.nextFloat() * (dcupperBound - dclowerBound);
  float theta0 = theta0lowerBound + random.nextFloat() * (theta0upperBound - theta0lowerBound);
  //float dc = 0.1;
  //float theta0 = PI / 6;
  float thetaAupperBound = min(theta0plusthetaAupperBound - theta0, theta0 - theta0minusthetaAlowerBound);
  float thetaA = thetaAlowerBound + random.nextFloat() * (thetaAupperBound - thetaAlowerBound);
  
  angles = rotate_angle(sim_length, period, thetaA, dc);
  body = new EllipseBody(xrandomFloat, yrandomFloat, n/24, 0.15, view);
  body2 = new EllipseBody(xrandomFloat, yrandomFloat, n/24, 0.15, view);
  float[] params = {xrandomFloat, yrandomFloat, theta0, thetaA, dc, n, n};
  body.rotate(thetaA + theta0);
  body2.rotate(-(thetaA + theta0));
  // angle_list = loadStrings(root_dir + "/angles/" + "sim_" + str(iter) + ".txt")[0];
  // angles = rotate_angle_discrete(sim_length, period, angle_list);
  bodyunion = new BodyUnion(body, body2);

  flow = new BDIM(n,n,1.,body);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);       //    and its legend
  data = new SaveVectorFieldFromBoundary(
    output_states_path + "sim_" + str(iter) + ".txt", 
    output_force_path + "sim_" + str(iter) + ".txt", 
    output_bdry_path + "sim_" + str(iter) + ".txt", 
    params,
    angles
  );  
}
void draw(){
  if ((t == 0.) && (iter < max_iter)){
    customsetup(iter);
  }
  if(t<stime){  // run simulation until t<Time
    bodyunion.follow();
    bodyunion.bodyList.get(0).rotate(flow.dt * angles[(int)t]); // !!!!!!! if Quick == False, use uniform flow.dt, but should be very small
    bodyunion.bodyList.get(1).rotate(-flow.dt * angles[(int)t]);
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    // flood.display(flow.u.curl());              // compute and display vorticity
    // bodyunion.display();   
    t+=flow.dt;
  }else if((stime<= t) && (t<etime)){  // run simulation until t<Time
    bodyunion.follow(); 
    bodyunion.bodyList.get(0).rotate(flow.dt * angles[(int)t]);
    bodyunion.bodyList.get(1).rotate(-flow.dt * angles[(int)t]);                 
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    bodyunion.display();     
    if (t % sampling_step==0){
      data.addField(flow.u, flow.p, t);
      data.addForce(bodyunion, flow.p);
      data.addBoundary(bodyunion);
    }
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();
    t = 0.;
    iter+=1;
    //exit();
  }
  if (max_iter <= iter){  // close and save everything when t>Time
    exit();
  }
}
