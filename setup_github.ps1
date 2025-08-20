# GitHub Repository Setup Script
# Run this after installing Git (https://git-scm.com/download/win)

# Navigate to project directory
cd "C:\Users\Opal\PycharmProjects\Ball_Detection"

# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "🎯 Initial commit: Real-time ball & laser detection system

✨ Features:
- Dual vision methods (OpenCV HSV + YOLOv8)
- Custom YOLO model training pipeline
- Interactive GUI with real-time parameter tuning
- Comprehensive educational documentation
- Multi-platform support (Windows/macOS/Linux)

📚 Educational focus with 50+ pages of learning materials
🧠 AI/ML demonstration with traditional vs modern approaches
🎮 Ready-to-run with webcam support"

# ✅ READY TO PUSH! Follow these steps:

echo "🎉 Git repository is initialized and committed!"
echo ""
echo "📝 NEXT STEPS:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: real-time-ball-laser-detection (recommended)"
echo "3. Description: Real-time object detection using OpenCV and YOLOv8 with educational focus"
echo "4. Make it Public"
echo "5. DO NOT initialize with README/license (we have them)"
echo "6. Click 'Create repository'"
echo ""
echo "7. After creating the repo, replace YOUR_USERNAME below and run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/real-time-ball-laser-detection.git"
echo "   git push -u origin main"
echo ""
echo "🔄 Current status: Ready to push $(git log --oneline | wc -l) commits with $(git ls-files | wc -l) files"
