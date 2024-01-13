class Instrument:

    def __init__(self, Symbol, Precision, TradeAmountStep):
        self.name = Symbol
        self.ins_type = "CURRENCY"
        self.displayName = Symbol
        self.pipLocation = pow(10, (Precision-1) * -1)
        self.tradeUnitsPrecision = Precision
        self.marginRate = 0.02
        self.displayPrecision = Precision
        self.TradeAmountStep = int(TradeAmountStep)

    def __repr__(self):
        return str(vars(self))

    @classmethod
    def FromApiObject(cls, ob):
        return Instrument(
            ob['Symbol'],
            ob['Precision'],
            ob['TradeAmountStep']
        )