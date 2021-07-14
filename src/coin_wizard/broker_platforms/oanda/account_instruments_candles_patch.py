#!/usr/bin/python3
"""Handle account endpoints."""
from oandapyV20.endpoints.apirequest import APIRequest
from oandapyV20.endpoints.decorators import dyndoc_insert, endpoint
from oandapyV20.endpoints.responses.accounts import responses
from abc import abstractmethod


class Accounts(APIRequest):
    """Accounts - class to handle the accounts endpoints."""

    ENDPOINT = ""
    METHOD = "GET"

    @abstractmethod
    @dyndoc_insert(responses)
    def __init__(self, accountID=None, instrument=None):
        """Instantiate an Accounts APIRequest instance.
        Parameters
        ----------
        accountID : string (optional)
            the accountID of the account. Optional when requesting
            all accounts. For all other requests to the endpoint it is
            required.

        instrument : string (optional)
        	A string containing the base currency and quote currency delimited by a “_”.
        """
        endpoint = self.ENDPOINT.format(accountID=accountID, instrument=instrument)
        super(Accounts, self).__init__(endpoint, method=self.METHOD)

@endpoint("v3/accounts/{accountID}/instruments/{instrument}/candles")
class AccountsInstrumentsCandles(Accounts):
    """AccountsInstrumentsCandles.
    Fetch candlestick data for an instrument.
    """

    @dyndoc_insert(responses)
    def __init__(self, accountID, instrument, params=None):
        """Instantiate an AccountInstruments request.
        Parameters
        ----------
        accountID : string (required)
            id of the account to perform the request on.
        instrument : string (required)
            name of the Instrument.
        params : dict (optional)
            query params to send, check developer.oanda.com for details.
        """
        super(AccountsInstrumentsCandles, self).__init__(accountID, instrument)
        self.params = params
