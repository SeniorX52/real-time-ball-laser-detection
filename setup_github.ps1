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

# Add GitHub remote (replace YOUR_USERNAME and CHOSEN_REPO_NAME)
# git remote add origin https://github.com/YOUR_USERNAME/CHOSEN_REPO_NAME.git

# Push to GitHub (after creating the repository on GitHub)
# git branch -M main
# git push -u origin main

echo "✅ Git repository initialized!"
echo "📝 Next steps:"
echo "1. Install Git if not already installed: https://git-scm.com/download/win"
echo "2. Create a new repository on GitHub with one of the suggested names"
echo "3. Replace YOUR_USERNAME and CHOSEN_REPO_NAME in this script"
echo "4. Uncomment and run the final git commands"
