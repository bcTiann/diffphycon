/**********************************
 SaveVectorField class
 
Saves the velocity and pressure field to a text file with customizable header
These files can be quite large!

example code:


SaveVectorField data;
AudreyTest test;

void setup(){
  int resolution = 128, xLengths=6, yLengths=3, zoom = 1;
  float xStart = -4, yDist =0.2;
  test = new AudreyTest(resolution, xLengths, yLengths, xStart , yDist, zoom);
  test.update();

  data = new SaveVectorField("saved/data.txt",test.body.a.coords,test.Re,resolution, test.n,test.m);
  data.addField(test.flow.u,test.flow.p);
  data.finish();
}
***********************************/
// class SaveVectorFieldForEllipse {
//   PrintWriter output;
//   PrintWriter output1;
//   int m, n;

//   SaveVectorFieldForEllipse(String name, float x, float y, float h, float a, float pivot, int n, int m, int iteration) {
//     this.m = m;
//     this.n = n;
//     output = createWriter(name);
//     output.println("%% Initial parameters for elliptic cylinder.");
//     output.print("; x = "+ x);
//     output.print("; y = "+ y);
//     output.print("; h = "+ h);
//     output.print("; a = "+ a);
//     output.print("; pivot = "+ pivot);
//     output.print("; n = "+ n);
//     output.print("; m = "+ m);
//     output.println(";");
    
//     output1 = createWriter("/Users/weilong/data/evaluation/force/sim_"+str(iteration)+".txt");
//   }


//   void addField(VectorField u, Field p) {
//     for (int j=1; j<m-1; j++) {
//       output.print("x-coords ");
//       for (int i=1; i<n-1; i++) {
//         output.print(u.x.a[i][j] +" ");
//       }
//       output.println(";");
//     }
//     for (int j=1; j<m-1; j++) {
//       output.print("y-coords ");
//       for (int i=1; i<n-1; i++) {
//         output.print(u.y.a[i][j] +" ");
//       }
//       output.println(";");
//     }
//     for (int j=1; j<m-1; j++) {
//       output.print("pressure ");
//       for (int i=1; i<n-1; i++) {
//         output.print(p.a[i][j] +" ");
//       }
//       output.println(";");
//     }
//   }

//   void addForce(BodyUnion bodyunion, Field p) {
//     output1.print("(x-force, y-force): ");
//     for (int k=0; k < bodyunion.bodyList.size(); k++)  {
//       output1.print(bodyunion.bodyList.get(k).pressForce(p) +" ");
//       output1.print(", ");
//     }
//     output1.println(" ;; ");
//   }
  
//   void finish() {
//     output.flush(); // Writes the remaining data to the file
//     output.close(); // Closes the file
//     output1.flush(); // Writes the remaining data to the file
//     output1.close(); // Closes the file
//   }
// } 


class SaveVectorFieldFromBoundary {
  PrintWriter output;
  PrintWriter output1;
  PrintWriter output2;
  int m, n;

  // SaveVectorFieldFromBoundary(String sim_path, String force_path, String bdry_path, int n, int m) {
  SaveVectorFieldFromBoundary(String sim_path, String force_path, String bdry_path, float[] params, float[] angles) {
    this.m = (int)params[5];
    this.n = (int)params[6];
    output = createWriter(sim_path);
    output.println("%% Initial parameters.");
    output.print("; n = "+ n);
    output.print("; m = "+ m);
    output.print("; x_start = "+ params[0]);
    output.print("; y_start = "+ params[1]);
    output.print("; theta0 in degree = "+ (params[2] * 180 / PI));
    output.print("; thetaA in degree = "+ (params[3] * 180 / PI));
    output.print("; dc = "+ params[4]);
    output.print("; angles_velocity = ");
    for (int i = 0; i < angles.length; i++) {
      output.print(angles[i] + ", ");
    }
    output.println(";");
    output1 = createWriter(force_path);
    output2 = createWriter(bdry_path);
  }


  void addField(VectorField u, Field p, float t) {
    for (int j=1; j<m-1; j++) {
      output.print("x-coords ");
      for (int i=1; i<n-1; i++) {
        output.print(u.x.a[i][j] +" ");
      }
      output.println(";");
    }
    for (int j=1; j<m-1; j++) {
      output.print("y-coords ");
      for (int i=1; i<n-1; i++) {
        output.print(u.y.a[i][j] +" ");
      }
      output.println(";");
    }
    for (int j=1; j<m-1; j++) {
      output.print("pressure " + t + " ");
      for (int i=1; i<n-1; i++) {
        output.print(p.a[i][j] +" ");
      }
      output.println(";");
    }
  }

  void addForce(BodyUnion bodyunion, Field p) {
    output1.print("(x-force, y-force): ");
    for (int k=0; k < bodyunion.bodyList.size(); k++)  {
      output1.print(bodyunion.bodyList.get(k).pressForce(p) +" ");
      output1.print(", ");
    }
    output1.println(" ;; ");
  }

  void addBoundary(BodyUnion bodyunion) {
    for (int k=0; k < bodyunion.bodyList.size(); k++)  {
      output2.print("boundary " + k + ": ");
      output2.println(bodyunion.bodyList.get(k).coords +" ");
    }
  }

  void finish() {
    output.flush(); // Writes the remaining data to the file
    output.close(); // Closes the file
    output1.flush(); // Writes the remaining data to the file
    output1.close(); // Closes the file
    output2.flush(); // Writes the remaining data to the file
    output2.close(); // Closes the file
  }
} 