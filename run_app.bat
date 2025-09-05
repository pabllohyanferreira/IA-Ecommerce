@echo off
echo ========================================
echo   E-commerce Analytics - Big Data & IA
echo ========================================
echo.
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo Gerando dataset de e-commerce...
python src/data_generator.py
echo.
echo Iniciando aplicacao...
echo A aplicacao sera aberta em: http://localhost:8501
echo.
streamlit run app.py
pause
