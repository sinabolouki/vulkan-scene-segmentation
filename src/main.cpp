#include "application.h"

int main() {
    Application app;

    if (!app.init(1280, 720, "Depth Peeling Renderer")) {
        return -1;
    }

    app.run();
    app.cleanup();

    return 0;
}