# SmartPantry 🥘

An intelligent kitchen management application that uses AI to help you track your pantry, reduce food waste, and discover delicious recipes based on what you have available.

## 🌟 Features

### AI-Powered Features
- **Photo Recognition**: Upload photos of your pantry items and let AI automatically identify and add them
- **Smart Recipe Generation**: Get personalized recipe suggestions based on your available ingredients
- **Quantity-Aware Recipes**: Recipes automatically scale based on your available quantities
- **Expiration Tracking**: Prioritize recipes for items expiring soon

### Pantry Management
- **Real-time Updates**: Add, edit, and delete items with instant synchronization
- **Expiration Tracking**: Visual indicators for items expiring soon
- **Quantity Management**: Track quantities with flexible units (bottles, cans, grams, etc.)
- **Search & Filter**: Quickly find items in your pantry
- **Duplicate Detection**: Prevents adding duplicate items (case-insensitive)

### Cross-Platform Support
- **Web Application**: Responsive web interface built with Flask and Bootstrap
- **iOS Mobile App**: Native SwiftUI app for on-the-go pantry management
- **RESTful API**: Unified API for both web and mobile clients

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key (for AI features)
- Node.js (optional, for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd code
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

4. **Run the application**
   ```bash
   python api/app.py
   ```
   The app will be available at `http://localhost:5000`

### Deployment

#### Vercel (Serverless)
The app is configured for Vercel serverless deployment:
- Set environment variables in Vercel dashboard
- Deploy using `vercel.json` configuration

#### Render (Traditional)
- Set environment variables in Render dashboard
- Deploy using `Procfile`

## 📁 Project Structure

```
code/
├── api/
│   ├── app.py              # Main Flask application
│   ├── index.py            # Vercel entry point
│   ├── templates/          # Jinja2 HTML templates
│   │   ├── index.html      # Pantry dashboard
│   │   ├── suggest_recipe.html
│   │   └── recipe_detail.html
│   └── static/             # Static assets (CSS, images)
├── Smart_pantry/          # iOS mobile app
├── requirements.txt        # Python dependencies
├── vercel.json            # Vercel configuration
└── Procfile              # Render configuration
```

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - User login
- `GET /api/auth/verify` - Verify authentication status

### Pantry Management
- `GET /` - Pantry dashboard (web)
- `POST /api/pantry` - Add item to pantry
- `PUT /api/pantry/<item_id>` - Update pantry item
- `DELETE /api/pantry/<item_id>` - Delete pantry item
- `GET /api/pantry` - Get all pantry items

### Recipe Generation
- `GET /suggest` - Get recipe suggestions (web)
- `POST /api/recipes/suggest` - Get recipe suggestions (API)
- `GET /recipe/<recipe_name>` - View recipe details

### Photo Recognition
- `POST /upload_photo` - Upload photo for AI recognition (web)
- `POST /api/upload_photo` - Upload photo (API)

## 🛠️ Technology Stack

### Backend
- **Flask 3.1.2** - Web framework
- **OpenAI API** - AI-powered features (GPT-4 Vision, GPT-4)
- **Python 3.13** - Programming language

### Frontend (Web)
- **Bootstrap 5** - UI framework
- **Jinja2** - Template engine
- **Vanilla JavaScript** - Client-side interactivity

### Mobile
- **SwiftUI** - iOS app framework
- **Swift** - Programming language

### Infrastructure
- **Vercel** - Serverless deployment
- **Render** - Traditional hosting option

## 🔒 Security Features

- Session-based authentication
- Password hashing (SHA-256)
- Input validation and sanitization
- CORS support for cross-origin requests
- Secure session cookies

## 📊 Performance Optimizations

- **In-memory caching**: 5-second TTL for user and pantry data
- **Atomic file writes**: Prevents data corruption
- **Efficient data normalization**: Single-pass processing
- **Conditional logging**: Reduces overhead in production

## 🐛 Known Issues & Limitations

1. **Password Security**: Currently uses SHA-256 (should upgrade to bcrypt for production)
2. **File-based Storage**: JSON files may not scale for large user bases
3. **Serverless Limitations**: File storage is ephemeral on Vercel
4. **Race Conditions**: Concurrent updates may cause data inconsistencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- OpenAI for GPT-4 Vision and GPT-4 APIs
- Flask community for excellent documentation
- Bootstrap team for the UI framework
