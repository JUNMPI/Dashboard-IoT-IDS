# GIFs de Demostración - Dashboard IoT-IDS

Esta carpeta contiene las animaciones GIF que demuestran el funcionamiento del sistema.

## Lista de GIFs Requeridos

1. **demo_comparacion.gif** - Demostración completa de comparación de modelos (15-20s)
2. **demo_tiempo_real.gif** - Simulación en tiempo real con gráficos (20-25s)
3. **demo_analisis.gif** - Proceso completo de análisis de archivo CSV (15-20s)
4. **demo_navegacion.gif** - Navegación general por todas las páginas (20-25s)

## Especificaciones

- **Formato**: GIF
- **Resolución**: 1280×720 (recomendado)
- **FPS**: 15-20
- **Duración**: 10-30 segundos
- **Tamaño máximo**: 5 MB por GIF
- **Colores**: 128-256

## Herramientas Recomendadas

- **Windows**: ScreenToGif (https://www.screentogif.com/)
- **Linux**: Peek (https://github.com/phw/peek)
- **macOS**: Kap (https://getkap.co/)

## Optimización

```bash
# Usando gifsicle
gifsicle -O3 --colors 256 input.gif -o output.gif

# Reducir tamaño
gifsicle --resize-width 1280 -O3 --colors 128 input.gif -o output.gif
```

## Instrucciones

Ver [VISUAL_GUIDE.md](../../VISUAL_GUIDE.md) para scripts detallados de grabación.

## Uso en README

```markdown
![Demo Comparación](docs/gifs/demo_comparacion.gif)
```
