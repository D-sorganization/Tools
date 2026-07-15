echo "Checking TabBar.tsx for Object.fromEntries(TABS.map...)"
cat src/p1am_control_system/frontend/src/components/TabBar.tsx | grep -B 2 -A 3 "Object.fromEntries("
