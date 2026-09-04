def numerical_gradient(f, x, h=0.0001):
    """
    Sayisal türev (numerical derivative): f fonksiyonunun x noktasındaki
    eğimini, x'i çok küçük bir h kadar arttırıp f'in ne kadar değiştiğini
    ölçerek yaklaşık hesaplar.

    f'(x) ~ [f(x+h) - f(x)] / h
    """
    return (f(x + h) - f(x)) / h