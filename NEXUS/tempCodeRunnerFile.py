from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from deep_translator import GoogleTranslator
from twilio.rest import Client
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farm.db'

# --- IMAGE UPLOAD CONFIG ---
UPLOAD_FOLDER = 'static/product_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- TWILIO CONFIG ---
app.config['TWILIO_ACCOUNT_SID'] = 'ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 
app.config['TWILIO_AUTH_TOKEN'] = 'your_auth_token_here'
app.config['TWILIO_PHONE_NUMBER'] = '+15550000000'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    category = db.Column(db.String(50))
    location = db.Column(db.String(150), nullable=True, default='Not specified')
    image = db.Column(db.String(150), nullable=True, default='default.jpg')
    farmer_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    farmer = db.relationship('User', foreign_keys=[farmer_id])

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'))
    consumer_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    farmer_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    quantity = db.Column(db.Integer, nullable=False, default=1) 
    status = db.Column(db.String(50), default='Pending')
    date = db.Column(db.DateTime, default=datetime.utcnow)
    
    product = db.relationship('Product', backref='orders')
    consumer = db.relationship('User', foreign_keys=[consumer_id], backref='my_orders')
    farmer = db.relationship('User', foreign_keys=[farmer_id], backref='sales', overlaps="farmer")

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='activities')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML FORECASTING ENGINE ---
class MLForecaster:
    # FIXED: Added 'r' prefix to treat the backslashes literally and fix the SyntaxError
    def __init__(self, data_file=r'C:\Users\SWARNENDU DAS\Desktop\NEXUS\historical_data.csv'):
        self.data_file = data_file
        self.model_price = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_qty = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.prod_mapping = {}
        self.train_model()

    def train_model(self):
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file)
                df['date'] = pd.to_datetime(df['date'])
                df['month'] = df['date'].dt.month
                df['day_of_week'] = df['date'].dt.dayofweek
                
                df['prod_code'], uniques = pd.factorize(df['product_name'])
                self.prod_mapping = {name: i for i, name in enumerate(uniques)}
                
                X = df[['prod_code', 'month', 'day_of_week']]
                y_price = df['price_per_kg']
                y_qty = df['quantity_sold']

                self.model_price.fit(X, y_price)
                self.model_qty.fit(X, y_qty)
                self.is_trained = True
            except Exception as e:
                print(f"ML Training Error: {e}")

    def predict_today(self, product_name=None):
        if not self.is_trained or not self.prod_mapping:
            return []
        
        now = datetime.now()
        target_list = [product_name] if (product_name and product_name in self.prod_mapping) else list(self.prod_mapping.keys())[:3]
        
        results = []
        for name in target_list:
            if name in self.prod_mapping:
                code = self.prod_mapping[name]
                features = np.array([[code, now.month, now.weekday()]])
                price = self.model_price.predict(features)[0]
                qty = self.model_qty.predict(features)[0]
                results.append({'name': name, 'price': round(price, 2), 'qty': int(qty)})
        return results

forecaster = MLForecaster()

# --- CONTEXT PROCESSOR & TRANSLATION ---
@app.context_processor
def inject_globals():
    def translate_text(text):
        try:
            dest_lang = session.get('lang', 'en')
            if dest_lang == 'en': return text
            return GoogleTranslator(source='auto', target=dest_lang).translate(text)
        except: return text

    now = datetime.now()
    return dict(
        translate=translate_text,
        ml_forecasts=forecaster.predict_today(),
        current_time=now.strftime("%I:%M %p"),
        current_date=now.strftime("%d %b, %Y"),
        current_day=now.strftime("%A")
    )

@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    session['lang'] = lang_code
    return redirect(request.referrer or url_for('index'))

# --- ROUTES ---
@app.route('/')
def index():
    products = Product.query.all()
    return render_template('index.html', products=products)

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/customer_service')
def customer_service(): return render_template('customer_service.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        phone = request.form.get('phone')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if User.query.filter((User.username == username) | (User.phone == phone)).first():
            flash('Username or Phone number already exists.')
            return redirect(url_for('register'))
            
        new_user = User(username=username, phone=phone, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        db.session.add(ActivityLog(user_id=new_user.id, action=f'Registered as {role}'))
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            db.session.add(ActivityLog(user_id=user.id, action='Logged in'))
            db.session.commit()
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'farmer':
        my_products = Product.query.filter_by(farmer_id=current_user.id).all()
        incoming_orders = Order.query.filter_by(farmer_id=current_user.id).all()
        total_sales = Order.query.filter_by(farmer_id=current_user.id, status='Accepted').count()
        return render_template('farmer_dashboard.html', products=my_products, orders=incoming_orders, sales=total_sales)
    else:
        search_query = request.args.get('q', '')
        if search_query:
            products = Product.query.filter(
                (Product.name.ilike(f'%{search_query}%')) |
                (Product.category.ilike(f'%{search_query}%')) |
                (Product.location.ilike(f'%{search_query}%'))
            ).all()
        else:
            products = Product.query.all()
        my_orders = Order.query.filter_by(consumer_id=current_user.id).all()
        return render_template('consumer_dashboard.html', products=products, orders=my_orders, search_query=search_query)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    users = User.query.all()
    products = Product.query.all()
    orders = Order.query.order_by(Order.date.desc()).all()
    logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(100).all()
    
    total_sales_kg = sum([o.quantity for o in orders if o.status == 'Accepted'])
    total_revenue = sum([(o.quantity * o.product.price) for o in orders if o.status == 'Accepted'])
    total_inventory = sum([p.quantity for p in products])
    
    return render_template('admin_dashboard.html', 
                           users=users, products=products, orders=orders, logs=logs,
                           total_sales_kg=total_sales_kg, total_revenue=total_revenue, total_inventory=total_inventory)

@app.route('/check_forecast', methods=['POST'])
@login_required
def check_forecast():
    product_name = request.form.get('product_check')
    results = forecaster.predict_today(product_name)
    my_products = Product.query.filter_by(farmer_id=current_user.id).all()
    incoming_orders = Order.query.filter_by(farmer_id=current_user.id).all()
    total_sales = Order.query.filter_by(farmer_id=current_user.id, status='Accepted').count()
    forecast_data = None
    if results:
        forecast_data = {'name': results[0]['name'], 'trend': 'ML Predicted', 'price': results[0]['price'], 'qty': results[0]['qty']}

    return render_template('farmer_dashboard.html', products=my_products, orders=incoming_orders, sales=total_sales, forecast_result=forecast_data)

@app.route('/add_product', methods=['POST'])
@login_required
def add_product():
    if current_user.role != 'farmer': return redirect(url_for('index'))
    name = request.form.get('name')
    price = float(request.form.get('price'))
    qty = int(request.form.get('quantity'))
    category = request.form.get('category')
    location = request.form.get('location')
    image_file = request.files.get('image')
    filename = 'default.jpg'
    
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image_file.filename}")
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    new_prod = Product(name=name, price=price, quantity=qty, category=category, location=location, image=filename, farmer_id=current_user.id)
    db.session.add(new_prod)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/buy/<int:product_id>', methods=['POST'])
@login_required
def buy_product(product_id):
    product = Product.query.get(product_id)
    order_qty = int(request.form.get('order_quantity', 1))
    if product and product.quantity >= order_qty:
        new_order = Order(product_id=product.id, consumer_id=current_user.id, farmer_id=product.farmer_id, quantity=order_qty)
        db.session.add(new_order)
        db.session.commit()
        flash(f'Order placed!')
    return redirect(url_for('dashboard'))

@app.route('/manage_order/<int:order_id>/<action>')
@login_required
def manage_order(order_id, action):
    order = Order.query.get(order_id)
    if not order or order.farmer_id != current_user.id: return "Unauthorized"
    if action == 'accept' and order.product.quantity >= order.quantity:
        order.status = 'Accepted'
        order.product.quantity -= order.quantity
        flash('Order Accepted!')
    elif action == 'reject': 
        order.status = 'Rejected'
        flash('Order Rejected.')
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='Subhajit Rudra').first():
            db.session.add(User(username='Subhajit Rudra', phone='Admin', password='Subhajit2005', role='admin'))
            db.session.commit()
    app.run(host='0.0.0.0', port=5000, debug=False)