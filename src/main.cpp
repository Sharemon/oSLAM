#include <iostream>
#include "DataLoader.hpp"

using namespace oSLAM;
using namespace cv;
using namespace std;


int main(int, char**) {

    DataLoader data_loader("/home/pxy/oSLAM/data/unit_test");
    Mat rgb, depth;
    bool ret;
    ret = data_loader.pop(rgb, depth);
    ret = data_loader.pop(rgb, depth);

    imwrite("./rgb.png", rgb);
    imwrite("./depth.png", depth);

    cout << "Hello, oSLAM!" << endl;

}
